
class Restorer():
    """
    This interface provides a way to retrieve an object
    from some storage like a file, a database or S3.
    """

    def restore(self):
        """
        :return: an object from some persisted store
        """
        raise NotImplementedError
