import os

try:
    # Python 2
    import cPickle as pickle
except ImportError:
    # Python 3
    import pickle


class BestCandidatePersistor(object):
    """
    A class which knows how to persist a given candidate as a best candidate.
    (That the candidate is "best" contributes to the naming of the file)
    """

    def __init__(self, experiment_dir, candidate_id):
        """
        Constructor.

        :param experiment_dir: The directory into which experiment results
                are written
        :param candidate_id: The id of the best candidate
        """

        self.experiment_dir = experiment_dir
        self.candidate_id = candidate_id
        self.file_name = os.path.join(experiment_dir,
                                      "best_chromo_i-%s" % candidate_id)

    def persist(self, candidate):
        """
        :param candidate: Writes the given best candidate as a pickle file
        """
        with open(self.file_name, 'wb') as f:
            pickle.dump(candidate, f)


    def restore(self):
        """
        :return: the best candidate read from a pickle file
        """
        best_candidate = None

        best_candidate_fp = self.file_name
        assert os.path.exists(best_candidate_fp)

        with open(best_candidate_fp) as f:
            best_candidate = pickle.load(f)

        return best_candidate

