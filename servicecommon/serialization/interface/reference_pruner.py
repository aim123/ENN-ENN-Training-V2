

class ReferencePruner():
    """
    An interface whose implementations know how to prune/graft repeat
    references to objects within an object hierarchy defined by a single root
    object.

    In an ideal persisted object hierarchy, there would already be no reference
    cycles and the notion of a ReferencePruner would not be needed at all.
    Strive to eliminate the need for ReferencePruners at all in your code.
    """

    def prune(self, obj):
        """
        :param obj: The object to be pruned
        :return: A copy of the provided object which has no repeat
                references in any of its referenced sub-objects.
        """
        raise NotImplementedError


    def graft(self, obj, graft_reference=None):
        """
        :param obj: The object to be grafted onto.
                    Can be None when no object is read in.
        :param graft_reference: the graft reference to be used for grafting
        :return: A copy of the provided object which has the repeat
                references restored in any of its referenced sub-objects.
        """
        raise NotImplementedError
