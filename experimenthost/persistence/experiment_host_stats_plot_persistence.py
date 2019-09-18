
from servicecommon.persistence.factory.persistence_factory \
    import PersistenceFactory
from servicecommon.persistence.interface.persistence \
    import Persistence
from servicecommon.persistence.mechanism.persistence_mechanisms \
    import PersistenceMechanisms
from servicecommon.serialization.format.serialization_formats \
    import SerializationFormats


class ExperimentHostStatsPlotPersistence(Persistence):
    """
    A class which knows how to persist plots of Experiment Host statistics
    to/from file(s).

    This class will produce a pretty JSON file that can be used to
    produce Server Stats plots from an experiment directory.
    The file itself is intended to be human-viewable.
    """

    def __init__(self, experiment_dir, extension, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param extension: string representing the image format
        :param logger: A logger to send messaging to
        """

        basename = "experiment_host_stats.{0}".format(extension)

        factory = PersistenceFactory(object_type="image",
                                     dictionary_converter=None,
                                     logger=logger)
        self.image_persistence = factory.create_persistence(experiment_dir,
                        basename,
                        persistence_mechanism=PersistenceMechanisms.LOCAL,
                        serialization_format=SerializationFormats.RAW_BYTES,
                        must_exist=False)


    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
                    In this implementation, we expect a dictionary.
        """

        self.image_persistence.persist(obj)


    def restore(self):
        """
        :return: an object from some persisted store.
        """

        image_bytes = self.image_persistence.restore()
        return image_bytes
