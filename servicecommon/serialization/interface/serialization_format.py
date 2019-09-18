
from servicecommon.serialization.interface.deserializer import Deserializer
from servicecommon.serialization.interface.serializer import Serializer
from servicecommon.serialization.interface.file_extension_provider \
    import FileExtensionProvider


class SerializationFormat(Serializer, Deserializer, FileExtensionProvider):
    """
    An interface which combines implementation aspects of a Serializer
    and a Deserializer with a format name for registration in a factory
    setting.
    """

    def from_object(self, obj):
        """
        :param obj: The object to serialize
        :return: an open file-like object for streaming the serialized
                bytes.  Any file cursors should be set to the beginning
                of the data (ala seek to the beginning).
        """
        raise NotImplementedError

    def to_object(self, fileobj):
        """
        :param fileobj: The file-like object to deserialize.
                It is expected that the file-like object be open
                and be pointing at the beginning of the data
                (ala seek to the beginning).

                After calling this method, the seek pointer
                will be at the end of the data. Closing of the
                fileobj is left to the caller.
        :return: the deserialized object
        """
        raise NotImplementedError


    def get_file_extension(self):
        """
        :return: A string representing a file extension for the
                serialization method, including the ".".
        """
        raise NotImplementedError
