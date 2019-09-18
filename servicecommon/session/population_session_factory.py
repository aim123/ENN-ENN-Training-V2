

class PopulationSessionFactory():
    """
    Abstract factory class for creating PopulationSession objects for
    communicating with a Population Service and/or underlying algorithm.
    """

    def create_session(self, population_service_host,
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
        raise NotImplementedError
