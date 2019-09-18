
import random

from framework.client_script.client_evaluator import ClientEvaluator


class RandomFitnessClientEvaluator(ClientEvaluator):
    """
    Dummy implementation of the ClientEvaluator that does
    no real domain evaluation and always returns a random fitness.
    """

    def evaluate(self, worker_request_dict, file_dict):
        """
        :param worker_request_dict: The Worker Request Dictionary
            as delivered by the Experiment Host
        :param file_dict: The File Dictionary,
            as delivered by the Experiment Host
        :return: A metrics dictionary containing measurements about
            the evaluation
        """

        metrics = {
            'term_criterion': 'TERM_EPOCH',
            'loss_history': [(1, 0.0)],
            'training_time': 0.0,
            'num_epochs_trained': 0,
            'total_num_epochs_trained': 0,
            'fitnesses': [(1, 0.0)],
            'avg_gpu_batch_time': [(1, 0.0)],
            'fitness': random.random()
        }

        return metrics
