
from framework.client_script.load_random_fitness_client_evaluator \
    import LoadRandomFitnessClientEvaluator
from framework.client_script.load_zero_fitness_client_evaluator \
    import LoadZeroFitnessClientEvaluator
from framework.client_script.random_fitness_client_evaluator \
    import RandomFitnessClientEvaluator
from framework.client_script.real_client_evaluator \
    import RealClientEvaluator
from framework.client_script.zero_fitness_client_evaluator \
    import ZeroFitnessClientEvaluator


class ClientEvaluatorFactory():
    """
    Factory class for ClientEvaluators.
    """

    def create_evaluator(self, name):

        evaluator = None

        use_name = name
        if name is None:
            use_name = "real"

        use_name = str(use_name)
        use_name = use_name.lower()

        if use_name.startswith("real"):
            evaluator = RealClientEvaluator()
        elif use_name.startswith("zero"):
            evaluator = ZeroFitnessClientEvaluator()
        elif use_name.startswith("random"):
            evaluator = RandomFitnessClientEvaluator()
        elif use_name.startswith("load_random") or \
            use_name.startswith("loadrandom"):
            evaluator = LoadRandomFitnessClientEvaluator()
        elif use_name.startswith("load_zero") or \
            use_name.startswith("loadzero"):
            evaluator = LoadZeroFitnessClientEvaluator()
        else:
            # Default
            print("Unknown client evaluator type {0}. Using Real".format(name))
            evaluator = RealClientEvaluator()

        return evaluator
