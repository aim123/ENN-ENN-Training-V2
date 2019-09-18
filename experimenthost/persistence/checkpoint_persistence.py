
from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence


class CheckpointPersistence(EasyJsonPersistence):
    """
    A class which knows how to persist lists of checkpoint ids that
    have been encountered during the course of a run
    """

    def __init__(self, folder=".", logger=None):
        super(CheckpointPersistence, self).__init__(
                    base_name="checkpoint_ids",
                    folder=folder,
                    logger=logger)

    def persist(self, obj):

        seen_checkpoint_ids = obj
        if not any(seen_checkpoint_ids):
            return

        ordered_no_dupes = self.remove_duplicates(seen_checkpoint_ids)

        persisted_dict = {
            "seen_checkpoint_ids": ordered_no_dupes
        }

        super(CheckpointPersistence, self).persist(persisted_dict)


    def restore(self):

        default_seen = []

        persisted_dict = super(CheckpointPersistence, self).restore()

        if persisted_dict is None:
            return default_seen

        seen_checkpoint_ids = persisted_dict.get("seen_checkpoint_ids",
                                                  default_seen)

        ordered_no_dupes = self.remove_duplicates(seen_checkpoint_ids)

        return ordered_no_dupes


    def remove_duplicates(self, seen_checkpoint_ids):
        """
        Remove duplicates from a given list of checkpoint ids
        """
        no_dupes_set = set()
        no_dupes = []
        for checkpoint_id in seen_checkpoint_ids:
            if checkpoint_id not in no_dupes_set:
                no_dupes_set.add(checkpoint_id)
                no_dupes.append(checkpoint_id)

        return no_dupes
