
import random

from experimenthost.tasks.evaluator_session_task import EvaluatorSessionTask


class RandomFitnessMockEvaluatorSessionTask(EvaluatorSessionTask):
    """
    Class which chooses random fitness for evalution.
    """

    def evaluate(self, population, verbose=False):
        """
        Mockup fitness based on the complexity of the individual
        and a random factor. Used in test instead of
        actual fitness based on performance of trained network
        """
        population_fitness_results = []
        for candidate in population:

            interpretation = candidate['interpretation']

            complexity = random.gauss(0.0, 1.0)

            fitness_results_dict = {'id': candidate['id'],
                                    'metrics': {
                                        'fitness': complexity,
                                        'alt_objective': complexity
                                    },
                                    'interpretation': interpretation}

            population_fitness_results.append(fitness_results_dict)

        return population_fitness_results
