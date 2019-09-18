

class ShutdownTask():
    """
    An interface for classes that need to shut down cleanly
    on an interrupt to implement to carry out their shutdown policy.

    Make sure implementations are registered with the SignalHandler
    in order for the shutdown policy to be executed at the appropriate time.
    """

    def shutdown(self, signum=None, frame=None):
        """
        :param signum: The signal number for signal package
        :param frame: The frame from the signal handler.
        """

        raise NotImplementedError
