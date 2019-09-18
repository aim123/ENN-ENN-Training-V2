
from experimenthost.util.shutdown_task import ShutdownTask


class CompletionServiceShutdownTask(ShutdownTask):
    """
    Shuts down the studio ml completion service.
    """

    def __init__(self, completion_service):
        self.completion_service = completion_service
        self.cs_exited = False

    def shutdown(self, signum=None, frame=None):
        """
        ShutdownTask interface fulfillment
        """
        if self.cs_exited:
            return

        self.cs_exited = True

        print("Shutting down completion service...")

        self.completion_service.__exit__()

        print("Shutdown completion service with following signal: {0}".format(
                signum))
