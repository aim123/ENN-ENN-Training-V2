
from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence


class MasterConfigPersistence(EasyJsonPersistence):
    """
    A class which knows how to persist a master config dict to/from file(s).

    This class will produce a pretty JSON file that can be used to
    examine the combined configuration that is used for an experiment
    for debugging purposes.

    The file itself is intended to be human-readable as well as
    machine-readable.
    """

    def __init__(self, experiment_dir, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param logger: A logger to send messaging to
        """

        super(MasterConfigPersistence, self).__init__(
                            base_name="master_config",
                            folder=experiment_dir,
                            logger=logger)
