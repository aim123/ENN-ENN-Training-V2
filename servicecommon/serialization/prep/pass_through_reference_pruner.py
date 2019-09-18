
from servicecommon.serialization.interface.reference_pruner \
    import ReferencePruner


class PassThroughReferencePruner(ReferencePruner):
    """
    A ReferencePruner implementation where nothing is pruned or grafted.
    """

    def prune(self, obj):
        """
        :param obj: The object to be pruned
        :return: A copy of the provided object which has no repeat
                references in any of its referenced sub-objects.
        """
        return obj

    def graft(self, obj, graft_reference=None):
        """
        :param obj: The object to be grafted onto
        :param graft_reference: the graft reference to be used for grafting
        :return: A copy of the provided object which has the repeat
                references restored in any of its referenced sub-objects.
        """
        return obj
