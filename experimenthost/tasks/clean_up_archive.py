
import glob
import os

from experimenthost.util.candidate_util import CandidateUtil
from framework.util.experiment_filer import ExperimentFiler

class CleanUpArchive():
    """
    Class to assist with cleaning up persisted weights.
    """

    def __init__(self, experiment_dir):
        self.filer = ExperimentFiler(experiment_dir)

    def clean_up(self, population):
        """
        Removes persisted weights of individuals that are
        no longer in the population.
        """
        base_path = self.filer.experiment_file("archive")
        candidate_util = CandidateUtil()
        candidate_ids = [candidate_util.get_candidate_id(candidate) \
                            for candidate in population]

        # XXX Impenetrable!
        for filepath in glob.glob(base_path + "/*"):
            file_id = (filepath.split('/')[-1]).split('.')[0]
            file_type = (filepath.split('/')[-1]).split('.')[1]
            if file_id not in candidate_ids and file_type == 'h5a':
                os.remove(filepath)
