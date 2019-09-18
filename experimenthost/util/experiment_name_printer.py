
from __future__ import print_function

import sys

from experimenthost.util.shutdown_task import ShutdownTask


class ExperimentNamePrinter(ShutdownTask):
    """
    Spits out experiment name
    """

    def __init__(self, experiment_name):
        """
        :param experiment_name: The name of the experiment to print
        """
        self.experiment_name = experiment_name


    def shutdown(self, signum=None, frame=None):
        """
        Called from signal handler.
        """
        self.print_experiment_name()


    def print_experiment_name(self):

        print()
        print("To reference this experiment again, make sure this is on the ")
        print("command line:")
        print("  --experiment_name={0}".format(self.experiment_name))
        print()
        sys.stdout.flush()
