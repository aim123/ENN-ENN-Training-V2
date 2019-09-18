
from framework.client_script.client_evaluator import ClientEvaluator
from framework.client_script.evaluator_loader import EvaluatorLoader

from framework.evaluator.data_pathdict import generate_data_pathdict


class RealClientEvaluator(ClientEvaluator):
    """
    An implementation of the ClientEvaluator interface that does
    real evaluation for the domain in the Worker.
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

        # Get an instance of the Model Evaluator
        loader = EvaluatorLoader()
        evaluator_instance = loader.load_evaluator(worker_request_dict)

        config = worker_request_dict.get('config', {})
        domain_config = config.get('domain_config', {})

        # Resolve the entries in the file_dict with actual data arrays
        data_path_dict = self.get_data_path_dict(domain_config, file_dict)
        data_dict = None
        if data_path_dict is not None:
            data_dict = evaluator_instance.load_data(domain_config, data_path_dict)

        # Evaluate the interpretation from the worker request
        candidate_id = worker_request_dict.get('id', None)
        interpretation = worker_request_dict.get('interpretation', {})
        metrics = evaluator_instance.evaluate_model(candidate_id,
                                    interpretation, domain_config, data_dict)

        return metrics


    def get_data_path_dict(self, domain_config, file_dict):

        # Check to see if data file paths are provided, if it is necessary
        # data is loaded using using generate_data_path_dict()
        data_path_dict = None
        if not domain_config.get('dummy_load', False):
            data_path_dict = file_dict
            if file_dict is None:
                data_path_dict = generate_data_pathdict(domain_config,
                                                    convert_urls=True)
        return data_path_dict
