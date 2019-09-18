
from servicecommon.persistence.interface.persistence \
    import Persistence


class NullPersistence(Persistence):
    """
    Null implementation of the Persistence interface.
    """

    def persist(self, obj):
        """
        Persists object passed in.

        :param obj: an object to be persisted
        """

    def restore(self):
        """
        :return: a restored instance of a previously persisted object
        """
        return None
