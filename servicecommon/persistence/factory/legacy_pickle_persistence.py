
from servicecommon.persistence.factory.abstract_persistence \
    import AbstractPersistence
from servicecommon.serialization.format.legacy_pickle_serialization_format \
    import LegacyPickleSerializationFormat


class LegacyPicklePersistence(AbstractPersistence):
    """
    Implementation of the AbstractPersistence class which
    saves pickled data of an object via some persistence mechanism.
    """

    def __init__(self, persistence_mechanism):
        """
        Constructor

        :param persistence_mechanism: the PersistenceMechanism to use
                for storage
        """

        super(LegacyPicklePersistence, self).__init__(persistence_mechanism)
        self._serialization = LegacyPickleSerializationFormat(
                                    persistence_mechanism.folder,
                                    persistence_mechanism.base_name,
                                    must_exist=persistence_mechanism.must_exist())

    def get_serialization_format(self):
        return self._serialization
