
from servicecommon.persistence.factory.abstract_persistence \
    import AbstractPersistence
from servicecommon.serialization.format.raw_bytes_serialization_format \
    import RawBytesSerializationFormat


class RawBytesPersistence(AbstractPersistence):
    """
    Implementation of the AbstractPersistence class which
    saves raw bytes data of an object via some persistence mechanism.
    """

    def __init__(self, persistence_mechanism):
        """
        Constructor

        :param persistence_mechanism: the PersistenceMechanism to use
                for storage
        """

        super(RawBytesPersistence, self).__init__(persistence_mechanism)
        self._serialization = RawBytesSerializationFormat()

    def get_serialization_format(self):
        return self._serialization
