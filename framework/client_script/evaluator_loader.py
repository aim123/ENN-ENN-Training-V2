
from framework.resolver.evaluator_resolver import EvaluatorResolver


class EvaluatorLoader():
    """
    Class which contains common logic for creating a ModelEvaluator
    instance from the unpacked code. Various ClientEvaluator implementations
    use this guy.
    """

    def load_evaluator(self, worker_request_dict):
        """
        :param worker_request_dict: The Worker Request Dictionary
            as delivered by the Experiment Host
        :return: An instance of the ModelEvaluator to use for evaluation.
        """

        config = worker_request_dict.get('config', {})
        domain_name = config.get('domain', None)
        domain_config = config.get('domain_config', {})

        if domain_name is None:
            raise ValueError("domain_name is None: Request:\n {0} ".format(
                                    str(worker_request_dict)))

        old_school_evaluator_name = config.get('evaluator', None)
        resolver = EvaluatorResolver()
        evaluator_class = resolver.resolve(domain_name,
                    class_name=domain_config.get('evaluator_class_name', None),
                    evaluator_name=old_school_evaluator_name,
                    extra_packages=config.get('extra_packages', None))
        evaluator_instance = evaluator_class()

        return evaluator_instance
