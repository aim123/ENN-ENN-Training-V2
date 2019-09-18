
from framework.serialization.metrics_dictionary_converter \
    import MetricsDictionaryConverter
from framework.util.generation_filer import GenerationFiler

from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence


class ResultsDictPersistence(EasyJsonPersistence):
    """
    A class which knows how to persist a results dict to/from file(s).

    The Results Dictionary is a dictionary mapping of id to candidate
    dictionary.  It is used in a few contexts:

        * Recovery from experiment stoppages
        * Experiment Visualization.

    This class will produce a pretty JSON file that can be used to
    produce/recover Results Dictionaries from a generation directory.
    The file itself is intended to be human-readable as well as
    machine-readable.
    """

    def __init__(self, experiment_dir, generation, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param generation: the generation number of the results dict
        :param logger: A logger to send messaging to
        """

        filer = GenerationFiler(experiment_dir, generation)
        generation_dir = filer.get_generation_dir()

        super(ResultsDictPersistence, self).__init__(
                    base_name="results_dict",
                    folder=generation_dir,
                    dictionary_converter=MetricsDictionaryConverter(
                                                    allow_restore_none=False),
                    logger=logger)
