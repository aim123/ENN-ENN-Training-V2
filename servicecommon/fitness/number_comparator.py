

from servicecommon.fitness.none_comparator import NoneComparator


class NumberComparator(NoneComparator):
    """
    An implementation of the Comparator interface for
    comparing two numbers.
    """

    def compare(self, obj1, obj2):
        """
        :param obj1: The first object offered for comparison
        :param obj2: The second object offered for comparison
        :return:  A negative integer, zero, or a positive integer as the first
                argument is less than, equal to, or greater than the second.
        """

        comparison = super(NumberComparator, self).compare(obj1, obj2)
        if comparison is not None:
            return comparison

        if obj1 < obj2:
            return -1
        if obj1 > obj2:
            return 1

        return 0
