
from experimenthost.completion_service.minio_shutdown_task \
    import MinioShutdownTask


class MinioStorageShutdownTask(MinioShutdownTask):
    """
    Removes files from the StudioML 'storage' minio server upon shutdown.

    Do the work of cleaning up the storage, given the state of the task.
    As of 11/2018, StudioML puts two directories under the storage bucket:

        1. experiments. We can use what we know to clean up completely here

        2. blobstore.   We do not have enough information at this level to
                        know what we can and cannot delete on a
                        per-experiment basis.
    """

    def __init__(self, config, studio_experiment_ids):
        """
        :param config: The completion_service config dictionary
        :param studio_experiment_ids: A list of known studio experiment ids.
        """

        folders = ["experiments/"]
        super(MinioStorageShutdownTask, self).__init__(config,
                                                       studio_experiment_ids,
                                                       config_key='storage',
                                                       folders=folders)
