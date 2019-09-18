
import time

import grpc

from servicecommon.service.python.generated \
    import population_structs_pb2 as service_messages

from servicecommon.session.checkpoint_policy import CheckpointPolicy
from servicecommon.session.extension_packaging import ExtensionPackaging
from servicecommon.session.grpc_client_retry import GrpcClientRetry
from servicecommon.session.population_response_packaging \
    import PopulationResponsePackaging
from servicecommon.session.population_session import PopulationSession
from servicecommon.session.service_routing import ServiceRouting


class SubmissionServicePopulationSession(PopulationSession):
    """
    An abstract Session implementation backed by a service which submits
    requests for new populations whose responses might initiate further
    requests that check if the population has been completed.

    This is abstract in the sense that the particulars of the GRPC
    implementation must be passed in, but since it is expected that
    the service is templated from the PopulationService GRPC definition,
    implementations talking to different services doling out different
    representations will have a lot of the same problems solved here.
    """

    # Tied for Public Enemy #8 for too-many-instance-attributes
    # pylint: disable=too-many-instance-attributes
    def __init__(self, host, port, timeout_in_seconds, logger,
                 service_name=None, service_stub=None, request_version=None):
        """
        Creates a SubmissionServicePopulationSession that connects to the
        SubmissionService and delegates its implementations to the service.

        :param host: the service host to connect to
        :param port: the service port
        :param timeout_in_seconds: timeout to use when communicating
                with the service
        :param logger: A logger to send messaging to
        :param service_name: the name of the service
        :param service_stub: the GRPC stub of the service which was
                templated from PopulationService
        :param request_version: the value for each request's version field
        """

        # This version corresponds to a vague notion of cluster version
        # that we want to talk to.  This doesn't necessarily correspond to
        # any version related to the internal implementation of that cluster.
        # It is populated as a field with each request so that some other entity
        # can potentially route or even provide some backwards compatibility
        # given this client's expectations.
        self.request_version = request_version
        self.service_stub = service_stub
        self.name = service_name

        self.logger = logger

        self.timeout_in_seconds = timeout_in_seconds
        self.poll_interval_seconds = 15

        # A timeout after which a new request will be issued.
        # This will prevent experiments from having to be restarted manually
        # when there are problems with the service itself, assuming the
        # service itself will lose the initial/next population request when
        # it is eventually kicked/repaired.
        hours = 60 * 60  # in seconds
        self.new_request_after_seconds = 1 * hours

        host = self._maybe_do_client_side_version_routing(host)

        # Set up a GRPC retry context that assumes that the initial submission
        # (either a GetInitialPopulation or a NextPopulation request)
        # has gone through on the service end when the UNAVAILABLE message
        # is first returned from sending the message *after* the connection
        # is first established.
        #
        # This happens quite often when large populations are submitted and
        # there is not enough time for the initial message to come back before
        # a connection timeout.  Assuming the message submission has gone
        # through upon receipt of UNAVAILABLE allows a single message to
        # percolate through the system, instead of flooding the system
        # with repeated attempts for GetInitial/NextPopulation until some
        # response is heard, which would generate waaay more work in the
        # system than is really necessary.
        #
        # There is the off chance, however, that the assumption of the message
        # going through is faulty.  This case should be picked up by the
        # fact that there is now a larger hour(s)-long retry cycle that resends
        # the initial request when the polling via GetPopulation yields
        # nothing after too long a time.
        limited_retry_set = set()
        limited_retry_set.add(grpc.StatusCode.UNAVAILABLE)

        self.initial_submission_retry = GrpcClientRetry(
            service_name=self.name,
            service_stub=self.service_stub,
            logger=self.logger,
            host=host,
            port=port,
            timeout_in_seconds=timeout_in_seconds,
            poll_interval_seconds=self.poll_interval_seconds,
            limited_retry_set=limited_retry_set,
            limited_retry_attempts=1)

        # Also set up a retry for the GetPopulation polling which is
        # done after the initial submission of a GetInitial/NextPopulation
        # request.  This request does not generate any subsequent work, so
        # it's OK to beat on the service with this request.
        self.polling_retry = GrpcClientRetry(
            service_name=self.name,
            service_stub=self.service_stub,
            logger=self.logger,
            host=host,
            port=port,
            timeout_in_seconds=timeout_in_seconds,
            poll_interval_seconds=self.poll_interval_seconds)

        self.extension_packaging = ExtensionPackaging()
        self.population_response_packaging = PopulationResponsePackaging()
        self.checkpoint_policy = CheckpointPolicy()


    def next_population(self, experiment_id, config,
                        evaluated_population_response):
        """
        :param experiment_id: A relatively unique human-readable String
                used for isolating artifacts of separate experiments

        :param config: Config instance

        :param evaluated_population_response: A population response containing
            the population for which the nextion will be based.
            If this is None, it is assumed that a new population will be started
            from scratch.

        :return: A PopulationResponse containing unevaluated candidates
            for the next generation.
        """
        self.logger.debug("Calling next_population() on the {0}",
                          self.name)

        # Create a request
        request = service_messages.PopulationRequest()
        request.version = self.request_version

        request.experiment_id = experiment_id
        request.config = self.extension_packaging.to_extension_bytes(config)
        population_response = \
            self.population_response_packaging.to_population_response(
                evaluated_population_response)

        # Protobufs does not allow to settting a message as a field.
        if population_response is None:
            #pylint: disable=line-too-long
            # Per https://stackoverflow.com/questions/29643295/how-to-set-a-protobuf-field-which-is-an-empty-message-in-python
            #pylint: disable=no-member
            request.evaluated_population_response.SetInParent()
        else:
            request.evaluated_population_response.CopyFrom(population_response) #pylint: disable=no-member

        # Submit the request
        rpc_method_args = [request]

        # Determine the next checkpoint id
        prev_checkpoint_id = None

        #pylint: disable=no-member
        if request.evaluated_population_response is not None:
            #pylint: disable=no-member
            prev_checkpoint_id = \
                request.evaluated_population_response.checkpoint_id
        next_checkpoint_id = \
            self.checkpoint_policy.next_checkpoint(prev_checkpoint_id)

        population_response_dict = self._poll_for_population(
            experiment_id,
            next_checkpoint_id,
            "NextPopulation",
            SubmissionServicePopulationSession._next_population_from_stub,
            rpc_method_args,
            give_up_seconds=self.new_request_after_seconds)

        self.logger.debug("Successfully called next_population().")
        return population_response_dict


    def get_population(self, experiment_id, checkpoint_id):
        """
        :param experiment_id: A String unique to the experiment
                used for isolating artifacts of separate experiments

        :param checkpoint_id: String specified of the checkpoint desired.

        :return: A PopulationResponse where ...

                "population" is a list of Chromosome data, previously
                generated by the implementation, comprising the entire
                population *that has yet to be evaluated* (not what you
                might have just evaluated). This means no previous fitness
                information can be expected from any component of the list.

                "generation_count" will be what was last handed out when
                the population was created

                "checkpoint_id" an actual not-None checkpoint_id
                            from which the returned unevaluated population
                            can be recovered via the get_population()
                            call.

                "evaluation_stats" will contain the same contents as were
                passed in when the population was created
        """

        self.logger.debug("Calling get_population() on the {0}",
                          self.name)

        # Create a request
        request = service_messages.ExistingPopulationRequest()
        request.version = self.request_version

        request.experiment_id = experiment_id
        request.checkpoint_id = checkpoint_id if checkpoint_id else ""

        # Submit the request
        rpc_method_args = [request]

        population_response_dict = self._poll_for_population(
            experiment_id,
            checkpoint_id,
            "GetPopulation",
            SubmissionServicePopulationSession._get_population_from_stub,
            rpc_method_args,
            give_up_seconds=self.new_request_after_seconds)


        self.logger.debug("Successfully called get_population().")
        return population_response_dict


    def _poll_for_population(self, experiment_id, checkpoint_id,
                             method_name, stub, rpc_method_args,
                             give_up_seconds=None):
        """
        Will call the given stub with the rpc_method_args and wait for a
        result with a valid population to come back.

        If the initial call fails for some reason (usually linux socket
        timeout inside the service), it assumes that it's going to take
        a while for the service to return with the population.  It switches
        to polling for a GetPopulation request given the future
        checkpoint id returned by the submission service.

        If after a long time these GetPopulation requests fail,
        it assumes there has been a problem with the service and attempts
        to retry the initial call, and the process starts all over again
        until a real answer comes back.

        :param experiment_id: The directory where checkpoint files go.
        :param checkpoint_id: a stab at what checkpoint id we should be
                looking for if we don't hear back from the request.
        :param method_name: GRPC protocol method to call
        :param stub: stub function to use to call the method
        :param rpc_method_args: a list (always) of arguments to pass to the
                   GRPC method
        :param give_up_seconds: number of seconds after which this method will
                    give up looking for a GetPopulation response
                    and retry the original GRPC method call.
        :return: a population response with valid population
        """

        population_response_dict = None
        while population_response_dict is None:

            response = None

            try:
                # Get the initial response from the service method.
                response = self.initial_submission_retry.must_have_response(
                    method_name, stub, *rpc_method_args)

            except KeyboardInterrupt as exception:
                # Allow for command-line quitting
                raise exception

            except Exception:
                self.logger.info("Assuming {0} request submitted.", method_name)
                # Otherwise pass

            # Read the initial response
            population_response_dict = {
                'checkpoint_id': checkpoint_id
            }
            if response is not None:
                population_response_dict = \
                    self.population_response_packaging.from_population_response(
                        response)

            # Determine when we should stop to retry not the
            # GetPopulation but the original request, in case the service
            # had problems and has come back to life.
            quit_seconds = None
            if give_up_seconds is not None:
                now_seconds = time.time()
                quit_seconds = now_seconds + give_up_seconds

            checkpoint_id = population_response_dict.get('checkpoint_id', None)
            population = population_response_dict.get('population', None)

            if population is None:
                population_response_dict = self._get_population_with_timeout(
                    experiment_id, checkpoint_id, quit_seconds)

            if population_response_dict is None:
                self.logger.info("No population yet. Retrying {0} request.",
                                 method_name)

        return population_response_dict


    def _get_population_with_timeout(self, experiment_id, checkpoint_id,
                                             quit_seconds=None):
        """
        Attempts to return the current population of the experiment
        given the checkpoint id... up to a time specified by quit_seconds.

        :param experiment_id: The directory where checkpoint files go.
        :param checkpoint_id: String specified of the checkpoint desired.
        :param quit_seconds: When the polling should stop. None implies no stop.
        :return: a population response dictionary with the current population of the
                experiment given the checkpoint or None if no population
                was returned just yet.
        """

        population_response_dict = self._raw_get_population(experiment_id,
                                                      checkpoint_id)

        population = None
        if population_response_dict is not None:
            # Update our test variables to see if we can exit the loop
            population = population_response_dict.get('population', None)

        while self._keep_polling(checkpoint_id, population, quit_seconds):

            self.logger.info("No population yet for {0} {1}. Waiting to retry.",
                  experiment_id, checkpoint_id)

            # Wait a little while to make the next request
            time.sleep(self.poll_interval_seconds)

            # The next request is the same as a request that resumes from
            # a checkpoint.
            population_response_dict = None
            try:
                population_response_dict = \
                    self._raw_get_population(experiment_id, checkpoint_id)

            except KeyboardInterrupt as exception:
                # Allow for command-line quitting
                raise exception

            except Exception as exception:
                self.logger.warning("Submission Service got error {}.", str(exception))
                self.logger.warning("Retrying until the service comes back up.")

            if population_response_dict is not None:
                # Update our test variables to see if we can exit the loop
                population = population_response_dict.get('population', None)

        # Caller expects the dictionary to be None if there is no population
        if population is None:
            population_response_dict = None

        return population_response_dict


    def _raw_get_population(self, experiment_id, checkpoint_id):
        """
        A single attempt at returning the current population of the experiment

        :param experiment_id: The directory where checkpoint files go.
        :param checkpoint_id: String specified of the checkpoint desired.
        :return: a population response dictionary with the current population of the
                experiment given the checkpoint.  This population could be
                None if the service has not finished preparing it.
        """
        self.logger.debug("Calling get_population() on the {0}",
                          self.name)

        # Create a request
        request = service_messages.ExistingPopulationRequest()
        request.version = self.request_version

        request.experiment_id = experiment_id
        request.checkpoint_id = checkpoint_id if checkpoint_id else ""

        # Submit the request
        rpc_method_args = [request]

        population_response_dict = None
        response = self.polling_retry.must_have_response(
            "GetPopulation",
            SubmissionServicePopulationSession._get_population_from_stub,
            *rpc_method_args)

        # Read the response
        if response is not None:
            population_response_dict = \
                self.population_response_packaging.from_population_response(
                    response)

        if population_response_dict is None:
            population_response_dict = { \
                'checkpoint_id': checkpoint_id,
                'population': None \
            }

        return population_response_dict


    def _keep_polling(self, checkpoint_id, population, quit_seconds):
        """
        :param checkpoint_id: String for the checkpoint desired.
        :param population: the value for the population response dictionary's population key
        :param quit_seconds: the time in seconds when polling should give up
        :return: True if the _poll_for_population() method who calls this
                should keep polling. False if it should give up.
        """

        # Actually, find reasons to stop polling

        # See if the time constraints will have us stop polling.
        now_seconds = time.time()
        keep_polling = (quit_seconds is None or now_seconds < quit_seconds)

        if keep_polling:
            # See if we have enough data to stop polling.
            # If the population response dictionary in the response has no population,
            # but does have a checkpoint id for future reference, keep
            # polling until the filling of the population is complete.
            keep_polling = (checkpoint_id is not None and population is None)

        return keep_polling


    def _maybe_do_client_side_version_routing(self, hostname):

        # Make sure there is some basis with which to modify the hostname
        service_routing = ServiceRouting()
        if service_routing.CLIENT_ROUTING_VERSION is None \
                or service_routing.CLIENT_ROUTING_VERSION == "":
            return hostname

        # Don't do anything to references used for local container testing
        if hostname == "localhost":
            return hostname

        # Split the hostname along its components
        components = hostname.split('.')

        # Append "-<CLIENT_ROUTING_VERSION>" to the first component
        first = components[0] + '-' + service_routing.CLIENT_ROUTING_VERSION
        components[0] = first

        # Reassemble
        new_hostname = ".".join(components)
        return new_hostname


    @classmethod
    def _next_population_from_stub(cls, stub, timeout_in_seconds, *args):
        """
        Global method associated with the session that calls NextPopulation
        given a grpc Stub already set up with a channel (socket) to call with.
        """
        response = stub.NextPopulation(*args, timeout=timeout_in_seconds)
        return response


    @classmethod
    def _get_population_from_stub(cls, stub, timeout_in_seconds, *args):
        """
        Global method associated with the session that calls GetPopulation
        given a grpc Stub already set up with a channel (socket) to call with.
        """
        response = stub.GetPopulation(*args, timeout=timeout_in_seconds)
        return response
