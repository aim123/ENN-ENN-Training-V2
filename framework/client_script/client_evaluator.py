

class ClientEvaluator():
    """
    Interface used by the client script to perform a particular style
    of evaluation.
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
        raise NotImplementedError
