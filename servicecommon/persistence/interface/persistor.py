
class Persistor():
    """
    This interface provides a way to save an object
    to some storage like a file, a database or S3.
    """

    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
        """
        raise NotImplementedError
