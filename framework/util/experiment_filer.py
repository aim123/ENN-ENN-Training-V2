
import os

class ExperimentFiler():
    """
    Class to handle creation of experiment file names
    """

    def __init__(self, experiment_dir):
        self.experiment_dir = experiment_dir

    def experiment_file(self, filename):
        exp_file = os.path.join(self.experiment_dir, filename)
        return exp_file
