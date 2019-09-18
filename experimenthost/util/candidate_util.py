
from servicecommon.fitness.candidate_metrics_provider \
    import CandidateMetricsProvider


class CandidateUtil():
    """
    Class for getting particular fields off a candidate.
    Storage might change, but these methods will provide
    consistent access as that happens.
    """

    def __init__(self, fitness_objectives=None):
        """
        Constructor.
        :param fitness_objectives: The FitnessObjectives object
        """
        self._fitness_objectives = fitness_objectives


    def get_candidate_id(self, candidate):
        """
        :return: the id of the candidate
        """
        return str(candidate['id'])


    def get_candidate_fitness(self, candidate, fitness_objective_index=0):
        """
        :param candidate: The candidate dictionary from which we want to get
                    fitness
        :param fitness_objective_index: The index of the fitness objective.
                By default this is 0, implying the primary fitness objective.
        :return: the fitness for the candidate
        """

        metrics_provider = CandidateMetricsProvider(candidate)
        fitness_value = \
            self._fitness_objectives.get_value_from_metrics_provider(
                                                    metrics_provider,
                                                    fitness_objective_index)
        return fitness_value
