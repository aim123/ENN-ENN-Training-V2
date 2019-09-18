

class Comparator():
    """
    An interface for comparing two objects.
    """

    def compare(self, obj1, obj2):
        """
        :param obj1: The first object offered for comparison
        :param obj2: The second object offered for comparison
        :return:  A negative integer, zero, or a positive integer as the first
                argument is less than, equal to, or greater than the second.
        """
        raise NotImplementedError
