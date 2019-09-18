
from servicecommon.service.python.generated \
    import enn_population_submission_service_pb2_grpc as service_stub

from servicecommon.session.submission_service_population_session \
    import SubmissionServicePopulationSession


class EnnSubmissionServicePopulationSession(SubmissionServicePopulationSession):

    """
    A Session implementation backed by a service which submits requests
    for new ENN populations whose responses might initiate further requests
    that check if the population has been completed.
    """

    def __init__(self, host, port, timeout_in_seconds, logger):
        """
        Creates a SubmissionServicePopulationSession that connects to the
        SubmissionService and delegates its implementations to the service.

        :param host: the service host to connect to
        :param port: the service port
        :param timeout_in_seconds: timeout ot use when communicating
                with the service
        :param logger: A logger to send messaging to
        """

        super(EnnSubmissionServicePopulationSession, self).__init__(host,
            port,
            timeout_in_seconds,
            service_name="ENN Population Submission Service",
            service_stub=service_stub.EnnPopulationSubmissionServiceStub,
            request_version="1.0.0",
            logger=logger)
