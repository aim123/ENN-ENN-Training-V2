
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


class EasyJsonPersistence(Persistence):
    """
    A superclass for concrete Persistence implementation needs
    where a dictionary is to be persisted in JSON format.
    A bunch of common defaults are set up and some common
    extra behaviors on persist() and restore() are implemented.
    """

    def __init__(self, base_name=None, folder=".", must_exist=False,
                 object_type="dict", dictionary_converter=None,
                 logger=None):
        """
        Constructor.

        :param base_name: The base name of the file.
                This does *not* include the ".json" extension.
        :param folder: The folder in which the file is to be persisted.
        :param must_exist: Default False.  When True, an error is
                raised when the file does not exist upon restore()
                When False, the lack of a file to restore from is
                ignored and a dictionary value of None is returned
        :param object_type: A string indicating the type of object to be
                persisted. "dict" by default.
        :param dictionary_converter: An implementation of a DictionaryConverter
                to use when converting the JSON to/from a dictionary.
                Default value of None implies that a
                PassThroughDictionaryConverter will be used, which does not
                modify the dictionary at all.
        :param logger: A logger to send messaging to
        """

        if base_name is None:
            raise ValueError("Must provide base_name in EasyJsonPersistence")

        # Set up the DictionaryConverter
        use_dictionary_converter = dictionary_converter
        if dictionary_converter is None:
            use_dictionary_converter = PassThroughDictionaryConverter()

        factory = PersistenceFactory(object_type=object_type,
                        dictionary_converter=use_dictionary_converter,
                        logger=logger)
        self.dict_persistence = factory.create_persistence(folder, base_name,
                        persistence_mechanism=PersistenceMechanisms.LOCAL,
                        serialization_format=SerializationFormats.JSON,
                        must_exist=must_exist)

    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
        """
        self.dict_persistence.persist(obj)


    def restore(self):
        """
        :return: an object from some persisted store as specified
                by the constructor.  If must_exist is False,
                this method can return None.
        """
        obj = self.dict_persistence.restore()
        return obj


    def get_file_reference(self):
        """
        :return: The full file reference of what is to be persisted
        """
        filename = self.dict_persistence.get_file_reference()
        return filename
