
import grpc

from servicecommon.session.enn_submission_service_population_session \
    import EnnSubmissionServicePopulationSession
from servicecommon.session.population_session_factory \
    import PopulationSessionFactory


class EnnPopulationSessionFactory(PopulationSessionFactory):
    """
    Stateless factory class for creating PopulationSession objects for
    communicating with the underlying algorithm.
    """

    def __init__(self, logger):
        """
        Constructor.

        :param logger: A logger to send messaging to
        """
        self.logger = logger


    def create_session(self,
                       population_service_host,
                       population_service_port,
                       timeout_in_seconds=None,
                       service=None):
        """
        :param population_service_host: The host name of for the
                    Population Service hosting the algorithm

        :param population_service_port: The port number for the
                    Population Service hosting the algorithm

        :param timeout_in_seconds: the timeout for each remote method call
                    If None, the timeout length is left to the implementation

        :param service: the name of the service to connect to

        :return: an appropriate PopulationSession instance based on the
                    arguments
        """

        session = None

        # Create the session: use the enn service if defined
        # otherwise default to DirectSession
        if population_service_host:

            try:
                if service is None or \
                    service == "EnnSubmissionPopulationService":

                    session = EnnSubmissionServicePopulationSession(
                                                    population_service_host,
                                                    population_service_port,
                                                    timeout_in_seconds,
                                                    self.logger)
                else:
                    from services.submissionservice.session.enn_service_population_session \
                        import EnnServicePopulationSession
                    session = EnnServicePopulationSession(
                                            population_service_host,
                                            population_service_port,
                                            timeout_in_seconds,
                                            self.logger)
            except grpc.FutureTimeoutError as exception:
                self.logger.error("Exception using {0}:{1} with service {2}",
                      population_service_host, population_service_port,
                      service)
                self.logger.error(exception)
                exit(1)

        else:
            # Import the DirectPopulationSession locally so people without
            # access to it can still use the PopulationSessionFactory class.
            from services.ennservice.session.candidate_direct_population_session \
                import CandidateDirectPopulationSession
            session = CandidateDirectPopulationSession(self.logger)

        return session
