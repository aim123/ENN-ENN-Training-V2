
from framework.serialization.metrics_dictionary_converter \
    import MetricsDictionaryConverter

from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence


class ExperimentHostStatsPersistence(EasyJsonPersistence):
    """
    A class which knows how to persist statistics the Experiment Host keeps
    to/from file(s).

    This class will produce a pretty JSON file that can be used to
    produce/recover stats Dictionaries from an experiment directory.
    The file itself is intended to be human-readable as well as
    machine-readable.
    """

    def __init__(self, experiment_dir, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param generation: the generation number of the results dict
        :param logger: The logger to send messaging to
        """

        super(ExperimentHostStatsPersistence, self).__init__(
                        base_name="experiment_host_stats",
                        folder=experiment_dir,
                        dictionary_converter=MetricsDictionaryConverter(
                                                    allow_restore_none=False),
                        logger=logger)
