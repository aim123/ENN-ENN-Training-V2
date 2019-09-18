
from servicecommon.session.extension_packaging import ExtensionPackaging
from servicecommon.service.python.generated \
    import population_structs_pb2 as service_messages


class PopulationResponsePackaging():
    """
    Class to assist in packaging up PopulationResponses suitable for protocol
    buffers transmission to/from an idiomatic python dictionary form.
    """

    def __init__(self, string_encoding='UTF-8'):
        """
        Constructor
        :param string_encoding: The string encoding to use when encoding/
            decoding strings.
        """
        self.extension_packaging = ExtensionPackaging(string_encoding)


    def to_population_response(self, response_dictionary,
                               default_checkpoint=None):
        """
        Convert a Population Response in python idiomatic dictionary form
        into a gRPC PopulationRequest structure suitable for transmission
        over the wire.
        :param response_dictionary: a dictionary set up to look like a
            PopulationResponse structure
        :param default_checkpoint: when None, returning of None is allowed
            as an entire response is allowed.
        :return: a PopulationResponse structure populated according to the
            fields of the response_dictionary
        """

        use_response_dict = response_dictionary
        if response_dictionary is None or \
            not isinstance(response_dictionary, dict) or \
            not bool(response_dictionary):

            if default_checkpoint is None:
                return None

            use_response_dict = {}

        # Always return some struct, but *not* None.
        # GRPC server infrastructure can't deal with None.
        population_response = service_messages.PopulationResponse()
        population_response.generation_count = \
            use_response_dict.get('generation_count', -1)
        population_response.checkpoint_id = \
            use_response_dict.get('checkpoint_id', default_checkpoint)

        evaluation_stats = \
            use_response_dict.get('evaluation_stats', None)
        evaluation_stats_bytes = \
            self.extension_packaging.to_extension_bytes(evaluation_stats)
        population_response.evaluation_stats = evaluation_stats_bytes

        population = None
        dict_population = use_response_dict.get('population', None)
        if dict_population is not None and isinstance(dict_population, list):
            population = []
            for candidate_dict in dict_population:
                candidate = service_messages.Candidate()
                candidate.id = candidate_dict.get('id', None)
                candidate.interpretation = \
                    self.extension_packaging.to_extension_bytes(
                        candidate_dict.get('interpretation', None))
                candidate.metrics = \
                    self.extension_packaging.to_extension_bytes(
                        candidate_dict.get('metrics', None))
                candidate.identity = \
                    self.extension_packaging.to_extension_bytes(
                        candidate_dict.get('identity', None))
                population.append(candidate)

        if population is not None and len(population) > 0:
            #pylint: disable=no-member
            population_response.population.extend(population)

        return population_response


    def from_population_response(self, population_response):
        """
        Convert a Population Response to its python idiomatic dictionary form
        :param population_response: a PopulationResponse structure handed
            over the wire
        :return: a dictionary set up to look like a PopulationResponse structure
            but all in dictionary form for internal pythonic consumption
            without regard to grpc as a communication mechanism
        """

        if population_response is None or \
            not isinstance(population_response,
                           service_messages.PopulationResponse):
            return None

        population = []
        for candidate in population_response.population:
            interpretation = \
                self.extension_packaging.from_extension_bytes(
                    candidate.interpretation)
            metrics = \
                self.extension_packaging.from_extension_bytes(
                    candidate.metrics)
            identity = \
                self.extension_packaging.from_extension_bytes(
                    candidate.identity)
            candidate = {
                "id" : candidate.id,
                "interpretation" : interpretation,
                "metrics" : metrics,
                "identity" : identity \
            }
            population.append(candidate)

        evaluation_stats = \
            self.extension_packaging.from_extension_bytes(
                population_response.evaluation_stats)

        # Check for an empty PopulationResponse message
        if evaluation_stats is None and \
            len(population) == 0 and \
            population_response.generation_count == 0 and \
            len(population_response.checkpoint_id) == 0:
            return None

        if len(population) == 0:
            population = None

        obj = { \
            "population": population,
            "generation_count": population_response.generation_count,
            "checkpoint_id": population_response.checkpoint_id,
            "evaluation_stats": evaluation_stats \
        }

        return obj
