
from experimenthost.completion_service.minio_shutdown_task \
    import MinioShutdownTask


class MinioDatabaseShutdownTask(MinioShutdownTask):
    """
    Removes files from the StudioML 'database' on the minio server
    upon shutdown.

    Do the work of cleaning up the database, given the state of the task.
    As of 11/2018, StudioML puts 3 directories under the database bucket:

        1. experiments. We can use what we know to clean up completely here
        2. projects. We can use what we know to clean up completely here
        3. users.   Unclear what to do here
    """

    def __init__(self, config, studio_experiment_ids):
        """
        :param config: The completion_service config dictionary
        :param studio_experiment_ids: A list of known studio experiment ids.
        """

        folders = ["experiments/", "projects/"]
        super(MinioDatabaseShutdownTask, self).__init__(config,
                                                       studio_experiment_ids,
                                                       config_key='database',
                                                       folders=folders)
