

from experimenthost.util.shutdown_task import ShutdownTask


class CompositeShutdownTask(ShutdownTask):
    """
    A ShutdownTask implementation that potentially calls multiple other
    ShutdownTasks.
    """

    def __init__(self):

        self.tasks = []

    def shutdown(self, signum=None, frame=None):
        """
        Called from signal handler.
        """
        self.do_shutdown(signum, frame)

    def append_shutdown_task(self, shutdown_task):
        """
        :param shutdown_task: the shutdown task to add at the end of the list
        """
        self.tasks.append(shutdown_task)


    def prepend_shutdown_task(self, shutdown_task):
        """
        :param shutdown_task: the shutdown task to add at the start of the list
        """
        self.tasks.insert(0, shutdown_task)

    def do_shutdown(self, signum, frame):
        """
        Actual shutdown process.
        """

        for task in self.tasks:
            task.shutdown(signum, frame)
