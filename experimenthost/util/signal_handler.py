
from __future__ import print_function

import signal
import traceback

from experimenthost.util.composite_shutdown_task import CompositeShutdownTask


class SignalHandler(CompositeShutdownTask):
    """
    Policy class for supporting clean shutdown across multiple ShutdownTasks.

    Processes using this to capture output with the linux "tee" command
    should consider using "tee -i" to allow tee to capture output upon ctrl-c.
    """

    def __init__(self):

        super(SignalHandler, self).__init__()

        self.exited = False

        self.orig_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self.shutdown)

        self.orig_sigterm_handler = signal.getsignal(signal.SIGTERM)
        signal.signal(signal.SIGTERM, self.shutdown)


    def shutdown(self, signum=None, frame=None):
        """
        Called from signal handler.
        """

        if self.exited:
            return

        try:
            self.exited = True

            self.do_shutdown(signum, frame)

            if signum is not None:
                if signum == signal.SIGINT:
                    signal.signal(signum, self.orig_sigint_handler)
                else:
                    assert signum == signal.SIGTERM
                    signal.signal(signum, self.orig_sigterm_handler)
        except Exception:
            print()
            print("Error: shutdown failed")
            print()
            traceback.print_exc()
