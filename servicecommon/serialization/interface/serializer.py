

class Serializer():
    """
    An interface which takes an object and serializes it into a particular
    serialized format.
    """

    def from_object(self, obj):
        """
        :param obj: The object to serialize
        :return: an open file-like object for streaming the serialized
                bytes.  Any file cursors should be set to the beginning
                of the data (ala seek to the beginning).
        """
        raise NotImplementedError
