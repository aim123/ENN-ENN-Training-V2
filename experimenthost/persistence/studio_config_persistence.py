
from servicecommon.persistence.factory.persistence_factory \
    import PersistenceFactory
from servicecommon.persistence.interface.persistence \
    import Persistence
from servicecommon.persistence.mechanism.persistence_mechanisms \
    import PersistenceMechanisms
from servicecommon.serialization.format.serialization_formats \
    import SerializationFormats
from servicecommon.serialization.prep.pass_through_dictionary_converter \
    import PassThroughDictionaryConverter


class StudioConfigPersistence(Persistence):
    """
    A class which knows how to persist a studio config dict to/from file(s).

    This class will produce a pretty YAML file that is used
    to save out the file that StudioML uses for its configuration.
    It can also be used to examine the combined configuration that
    is used for an experiment for debugging purposes.

    The file itself is intended to be human-readable as well as
    machine-readable.
    """

    def __init__(self, experiment_dir, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param logger: A logger to send messaging to
        """

        basename = "studio_config"

        dictionary_converter = PassThroughDictionaryConverter()
        factory = PersistenceFactory(object_type="dict",
                                     dictionary_converter=dictionary_converter,
                                     logger=logger)
        self.dict_persistence = factory.create_persistence(experiment_dir,
                        basename,
                        persistence_mechanism=PersistenceMechanisms.LOCAL,
                        serialization_format=SerializationFormats.YAML,
                        must_exist=False)


    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
                    In this implementation, we expect a dictionary.
        """
        self.dict_persistence.persist(obj)


    def restore(self):
        """
        :return: an object from some persisted store.
                If the file was not found we return an empty dictionary.
        """
        return self.dict_persistence.restore()


    def get_file_reference(self):
        """
        :return: The full file reference of what is to be persisted
        """
        return self.dict_persistence.get_file_reference()
