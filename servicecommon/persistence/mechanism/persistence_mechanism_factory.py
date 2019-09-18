
from servicecommon.persistence.mechanism.local_file_persistence_mechanism \
    import LocalFilePersistenceMechanism
from servicecommon.persistence.mechanism.persistence_mechanisms \
    import PersistenceMechanisms
from servicecommon.persistence.mechanism.s3_file_persistence_mechanism \
    import S3FilePersistenceMechanism


class PersistenceMechanismFactory():
    """
    Factory class for PersistenceMechanisms.
    Given:
    1. a string specifying persistence mechanism type
    2. a "folder" passed from the caller
    3. a "base_name" passed from the caller
            (i.e. file name without ".<extension>")

    ... the create_persistence_mechanism() method will dish out the correct
        PersistenceMechanism implementation.
    """

    def __init__(self, bucket_base="", key_base="", must_exist=True,
                 object_type="object", logger=None):
        """
        Constructor.

        :param bucket_base:  The bucket base for S3 storage
        :param key_base:  The key (folder) base for S3 storage
        :param must_exist: Default True.  When False, if the file does
                not exist upon restore() no exception is raised.
                When True, an exception is raised.
        :param object_type: A string describing what kind of object
                is to be persisted.
        :param logger: A logger to send messaging to
        """
        self.bucket_base = bucket_base
        self.key_base = key_base
        self.must_exist = must_exist
        self.object_type = object_type
        self.fallback = PersistenceMechanisms.NULL
        self.logger = logger


    def create_persistence_mechanism(self, folder, base_name,
                                        persistence_mechanism=None,
                                        must_exist=None):
        """
        :param folder: Directory/Folder of where the persisted
                    file should reside.
        :param base_name: File name for the persisted file.
        :param persistence_mechanism: a string description of the persistence
                mechanism desired.
        :param must_exist: Default None.  When False, if the file does
                not exist upon restore() no exception is raised.
                When True, an exception is raised.
        :return: a new PersistenceMechanism given all the specifications
        """

        use_must_exist = must_exist
        if must_exist is None:
            use_must_exist = self.must_exist

        use_persistence_mechanism = self._resolve_persistence_type(
                                            persistence_mechanism)

        persistence_mechanism_instance = None
        if use_persistence_mechanism is None or \
            use_persistence_mechanism.lower() == PersistenceMechanisms.NULL:
            persistence_mechanism_instance = None
        elif use_persistence_mechanism.lower() == PersistenceMechanisms.LOCAL:
            persistence_mechanism_instance = LocalFilePersistenceMechanism(
                            folder, base_name,
                            must_exist=use_must_exist)
        elif use_persistence_mechanism.lower() == PersistenceMechanisms.S3:
            persistence_mechanism_instance = S3FilePersistenceMechanism(
                            folder, base_name,
                            must_exist=use_must_exist,
                            bucket_base=self.bucket_base,
                            key_base=self.key_base)
        else:
            message = "Don't know persistence mechanism '{0}' for type '{1}'."
            if self.logger is not None:
                self.logger.warning(message,
                                    use_persistence_mechanism,
                                    self.object_type)
            persistence_mechanism_instance = None

        return persistence_mechanism_instance


    def _resolve_persistence_type(self, persistence_mechanism):
        """
        :param persistence_mechanism: a string description of the
                    persistence mechanism desired.
                    If None, use the persistence mechanism in the fallback
                    Otherwise, use the override in this argument
        :return: a tuple of (persistence mechanism, serialization format)
        """

        # Find the Persistence Mechanism type to use
        use_persistence_mechanism = self._find_persistence_mechanism(
                                            persistence_mechanism)
        if use_persistence_mechanism is None:
            # None found for argument, use fallback
            message = "Don't know persistence mechanism '{0}' for type '{1}'." + \
                        " Using fallback {2}."
            if self.logger is not None:
                self.logger.warning(message,
                                    persistence_mechanism,
                                    self.object_type,
                                    self.fallback)
            use_persistence_mechanism = self.fallback

        return use_persistence_mechanism


    def _find_persistence_mechanism(self, persistence_mechanism):
        """
        :param persistence_mechanism: The string name of the
                persistence mechanism to use.
        :return: The matching cannonical string for the persistence mechanism
                if it is found in the list of PERSISTENCE_MECHANISMS.
                None otherwise.
        """

        # Figure out the Persistence Mechanism specified in the argument
        found_persistence_mechanism = None
        if persistence_mechanism is not None:
            for mechanism in PersistenceMechanisms.PERSISTENCE_MECHANISMS:
                if persistence_mechanism.lower() == mechanism.lower():
                    found_persistence_mechanism = mechanism

        return found_persistence_mechanism
