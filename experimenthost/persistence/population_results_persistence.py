
from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence

from framework.util.generation_filer import GenerationFiler

from experimenthost.persistence.population_results_dictionary_converter \
    import PopulationResultsDictionaryConverter


class PopulationResultsPersistence(EasyJsonPersistence):
    """
    A class which knows how to persist a list of population results to
    a file.

    This class will produce a pretty-JSON file that can be used
    to inspect the information being sent to the ENN service.
    """

    def __init__(self, experiment_dir, generation, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param generation: the generation number of the population results
        :param logger: A logger to send messaging to
        """

        filer = GenerationFiler(experiment_dir, generation)
        generation_dir = filer.get_generation_dir()

        super(PopulationResultsPersistence, self).__init__(
                base_name="population_results",
                folder=generation_dir,
                dictionary_converter=PopulationResultsDictionaryConverter(),
                logger=logger)
