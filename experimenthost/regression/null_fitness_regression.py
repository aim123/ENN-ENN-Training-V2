
from experimenthost.regression.fitness_regression import FitnessRegression

class NullFitnessRegression(FitnessRegression):
    """
    Implementation of the FitnessRegression interface that does nothing.
    """

    def add_sample(self, id_key, metrics):
        """
        Adds a sample to the regression
        :param id_key: the id of the candidate
        :param metrics: the metrics for the candidate
        """
        return


    def update(self, worker_results_dict, evaluated_candidate_dict):
        """
        Updates per the fitness regression
        :param worker_results_dict: a map of id -> worker_responses
        :param evaluated_candidate_dict: a map of id -> evaluated_candidates
        """
        return
