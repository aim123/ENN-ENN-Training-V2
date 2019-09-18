
import os

from servicecommon.persistence.factory.json_gzip_persistence \
    import JsonGzipPersistence
from servicecommon.persistence.factory.json_persistence \
    import JsonPersistence
from servicecommon.persistence.factory.legacy_pickle_persistence \
    import LegacyPicklePersistence
from servicecommon.persistence.factory.raw_bytes_persistence \
    import RawBytesPersistence
from servicecommon.persistence.factory.text_persistence \
    import TextPersistence
from servicecommon.persistence.factory.null_persistence \
    import NullPersistence
from servicecommon.persistence.factory.yaml_persistence \
    import YamlPersistence

from servicecommon.persistence.mechanism.persistence_mechanism_factory \
    import PersistenceMechanismFactory


from servicecommon.serialization.format.serialization_formats \
    import SerializationFormats


class PersistenceFactory():
    """
    Factory class for Persistence implementations.
    Given:
    1. a string specifying PersistenceMechanism type
    2. a "persist_dir" passed from the caller (which often is experiment name)
    3. a "persist_file" passed from the caller (i.e. file name)

    ... the create_persistence() method will dish out the correct persistence
        implementation.
    """


    def __init__(self, bucket_base="", key_base="", object_type="object",
                 reference_pruner=None, dictionary_converter=None,
                 logger=None):
        """
        Constructor.

        :param bucket_base:  The bucket base for S3 storage
        :param key_base:  The key (folder) base for S3 storage
        :param object_type: A string describing what kind of object
                is to be persisted.
        :param reference_pruner: A ReferencePruner implementation
                to prevent persisting reference data twice.
        :param dictionary_converter: A DictionaryConverter implementation
                that knows how to convert an object to/from a data-only
                dictionary.
        :param logger: a logger to send messaging to
        """

        self.persistence_factory = PersistenceMechanismFactory(
                                        bucket_base=bucket_base,
                                        key_base=key_base,
                                        object_type=object_type,
                                        logger=logger)
        self.object_type = object_type
        self.reference_pruner = reference_pruner
        self.dictionary_converter = dictionary_converter
        self.fallback = SerializationFormats.JSON
        self.logger = logger


    def create_persistence(self, persist_dir, persist_file,
                         serialization_format=None,
                         persistence_mechanism=None,
                         must_exist=True):
        """
        :param persist_dir: Directory/Folder of where the persisted
                    file should reside.
        :param persist_file: File name for the persisted file.
        :param serialization_format: a string description of the
                SerializationFormat format desired.
        :param persistence_mechanism: a string description of the persistence
                mechanism desired.
        :param must_exist: When False, if the file does
                not exist upon restore() no exception is raised.
                When True (the default), an exception is raised.
        :return: a new Persistence implementation given all the specifications
        """

        use_serialization_format = self._resolve_serialization_format(
                                            serialization_format)

        use_persist_dir, use_persist_file = self._rearrange_components(
                                                    persist_dir, persist_file)

        persistence_mechanism_instance = \
            self.persistence_factory.create_persistence_mechanism(
                            use_persist_dir,
                            use_persist_file,
                            persistence_mechanism=persistence_mechanism,
                            must_exist=must_exist)

        persistence = None
        if persistence_mechanism_instance is None:
            persistence = NullPersistence()
        elif use_serialization_format == SerializationFormats.LEGACY_PICKLE:
            persistence = LegacyPicklePersistence(persistence_mechanism_instance)
        elif use_serialization_format == SerializationFormats.JSON:
            persistence = JsonPersistence(persistence_mechanism_instance,
                            reference_pruner=self.reference_pruner,
                            dictionary_converter=self.dictionary_converter,
                            logger=self.logger)
        elif use_serialization_format == SerializationFormats.JSON_GZIP:
            persistence = JsonGzipPersistence(persistence_mechanism_instance,
                            reference_pruner=self.reference_pruner,
                            dictionary_converter=self.dictionary_converter,
                            logger=self.logger)
        elif use_serialization_format == SerializationFormats.RAW_BYTES:
            persistence = RawBytesPersistence(persistence_mechanism_instance)
        elif use_serialization_format == SerializationFormats.TEXT:
            persistence = TextPersistence(persistence_mechanism_instance)
        elif use_serialization_format == SerializationFormats.YAML:
            persistence = YamlPersistence(persistence_mechanism_instance,
                            reference_pruner=self.reference_pruner,
                            dictionary_converter=self.dictionary_converter,
                            logger=self.logger)
        else:
            # Default
            message = "Don't know serialization format '{0}' for type '{1}'." +\
                        " Using fallback {2}."
            if self.logger is not None:
                self.logger.warning(message,
                                    use_serialization_format,
                                    self.object_type,
                                    self.fallback)
            persistence = self.create_persistence(use_persist_dir,
                            use_persist_file,
                            serialization_format=self.fallback,
                            persistence_mechanism=persistence_mechanism,
                            must_exist=must_exist)

        return persistence


    def _resolve_serialization_format(self, serialization_format):
        """
        :param serialization_format: a string description of the
                    serialization format desired.
                    If None, use the serialization format in the fallback
                    Otherwise, use the override in this argument
        :return: a string of the accepted serialization format
        """

        # Find the SerializationFormat type to use
        use_serialization_format = self._find_serialization_format(
                                            serialization_format)
        if use_serialization_format is None:
            # None found for argument, use fallback
            message = "Don't know serialization format '{0}' for type '{1}'." + \
                        " Using fallback {2}."
            if self.logger is not None:
                self.logger.warning(message,
                                    serialization_format,
                                    self.object_type,
                                    self.fallback)
            use_serialization_format = self.fallback

        return use_serialization_format


    def _find_serialization_format(self, serialization_format):
        """
        :param serialization_format: The string name of the
                serialization_format mechanism to use.
        :return: The matching cannonical string for the serialization format
                if it is found in the list of SERIALIZATION_FORMATS.
                None otherwise.
        """

        # Figure out the SerializationFormat specified in the fallback
        found_serialization_format = None
        if serialization_format is not None:
            for serialization in SerializationFormats.SERIALIZATION_FORMATS:
                if serialization_format.lower() == serialization.lower():
                    found_serialization_format = serialization

        return found_serialization_format


    def _rearrange_components(self, persist_dir, persist_file):
        """
        Potentially rearrange the components of the persistence path
        in case one has pieces of the other.

        :param persist_dir: Directory/Folder of where the persisted
                    file should reside.
        :param persist_file: File name for the persisted file.
        """
        use_persist_dir = persist_dir
        use_persist_file = persist_file

        # See if there are any directory components in persist_file
        split_tuple = os.path.split(persist_file)
        head = split_tuple[0]
        if head is not None:
            use_persist_dir = os.path.join(persist_dir, head)

        return use_persist_dir, use_persist_file
