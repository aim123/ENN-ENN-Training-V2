
from servicecommon.persistence.interface.persistor import Persistor
from servicecommon.persistence.interface.restorer import Restorer


class Persistence(Persistor, Restorer):
    """
    Interface which allows multiple mechanisms of persistence for an object.
    How and where entities are persisted are left as implementation details.
    """

    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
        """
        raise NotImplementedError

    def restore(self):
        """
        :return: an object from some persisted store
        """
        raise NotImplementedError
