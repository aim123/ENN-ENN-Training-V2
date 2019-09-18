
from framework.client_script.evaluator_loader import EvaluatorLoader
from framework.client_script.zero_fitness_client_evaluator \
    import ZeroFitnessClientEvaluator

class LoadZeroFitnessClientEvaluator(ZeroFitnessClientEvaluator):
    """
    Dummy implementation of the ClientEvaluator that does
    no real domain evaluation but does load the domain's ModelEvaluator
    and always returns 0 for fitness.
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

        loader = EvaluatorLoader()

        # We really do want to load the evaluator here.
        # This would tell us in tests that something is wrong.
        # Note: _ is pythonic for unused variable
        _ = loader.load_evaluator(worker_request_dict)

        return super(LoadZeroFitnessClientEvaluator, self).evaluate(
                    worker_request_dict, file_dict)
