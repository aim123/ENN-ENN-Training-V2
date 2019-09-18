
class CheckpointPolicy():
    """
    A move of the checkpoint policy out of the submission sessions
    and into its own place.  Ultimately this would not have to live
    on the client-side at all, but it's necessary for now until the
    Submission Service(s) change their ways.
    """

    def next_checkpoint(self, checkpoint_id):
        """
        :param checkpoint_id: The previous checkpoint id
        :return: the next checkpoint id
        """

        # We currently expect a string like "checkpoint_N"
        # where N is an integer.
        # XXX Not good that checkpoint naming policy is outside of
        #     service

        next_checkpoint_id = "checkpoint_0"
        if checkpoint_id is not None and len(checkpoint_id) > 0:
            check_split = checkpoint_id.split("_")
            gen = int(check_split[1]) + 1
            next_checkpoint_id = check_split[0] + "_" + str(gen)
        return next_checkpoint_id
