

from servicecommon.fitness.comparator import Comparator


class NoneComparator(Comparator):
    """
    An implementation of the Comparator interface for
    comparing two objects which might be None.
    """

    def compare(self, obj1, obj2):
        """
        :param obj1: The first object offered for comparison
        :param obj2: The second object offered for comparison
        :return:  0 if both objects are None.
                  -1 if obj1 is None only
                  1 if obj2 is None only
                  None if both are not None. This allows a return value
                  for a composite Comparator to know it has to do something.
        """

        if obj1 is None and obj2 is None:
            return 0

        if obj1 is None:
            return -1

        if obj2 is None:
            return 1

        return None
