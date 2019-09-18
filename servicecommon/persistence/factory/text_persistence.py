
from servicecommon.persistence.factory.abstract_persistence \
    import AbstractPersistence
from servicecommon.serialization.format.text_serialization_format \
    import TextSerializationFormat


class TextPersistence(AbstractPersistence):
    """
    Implementation of the AbstractPersistence class which
    saves text data of an object via some persistence mechanism.
    """

    def __init__(self, persistence_mechanism):
        """
        Constructor

        :param persistence_mechanism: the PersistenceMechanism to use
                for storage
        """

        super(TextPersistence, self).__init__(persistence_mechanism)
        self._serialization = TextSerializationFormat(
                                must_exist=persistence_mechanism.must_exist())

    def get_serialization_format(self):
        return self._serialization
