

class FitnessRegression():
    """
    Interface defining the calls for fitness regression
    to be done from the CompletionServiceEvaluatorSessionTask (currently).
    """

    def add_sample(self, id_key, metrics):
        """
        Adds a sample to the regression
        :param id_key: the id of the candidate
        :param metrics: the metrics for the candidate
        """
        raise NotImplementedError


    def update(self, worker_results_dict, evaluated_candidate_dict):
        """
        Updates per the fitness regression
        :param worker_results_dict: a map of id -> worker_responses
        :param evaluated_candidate_dict: a map of id -> evaluated_candidates
        """
        raise NotImplementedError
