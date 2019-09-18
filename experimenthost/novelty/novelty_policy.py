

class NoveltyPolicy():
    """
    Interface outlining methods to call for novelty policy.
    """

    def update(self, evaluated_candidate_dict):
        """
        Updates the novelty archive after all results come in.
        :param evaluated_candidate_dict: a mapping of id -> evaluated candidate
        """
        raise NotImplementedError

    def compute_novelty(self, metrics, default_value):
        """
        :param metrics: The metrics dictionary from an evaluated candidate
        :param default_value: default value if novelty cannot be computed
        :return: computed novelty measurement based on metrics.
        """
        raise NotImplementedError
