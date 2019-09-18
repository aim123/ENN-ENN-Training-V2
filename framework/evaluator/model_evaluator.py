

class ModelEvaluator():
    """
    An interface whose implementations know how to evaluate some evolved model.

    Main entry point is evaluate_model().
    """

    def load_data(self, domain_config, data_pathdict):
        """
        :param domain_config: The config dictionary describing the domain
                evaluation parameters
        :param data_pathdict: A dictionary of data files to use
        :return: a single dictionary whose keys describe domain-specific
                    data sets, and whose values are the data sets themselves
                    (often numpy arrays)
        """
        raise NotImplementedError


    def evaluate_model(self, candidate_id, interpretation, domain_config,
                        data_dict, model_weights=None):
        """
        Evaluate the given model interpretation.

        This is the main entry point for candidate evaluation,
        called by the client.py worker entry point script.

        :param candidate_id: the string identifier of the candidate to evaluate
        :param interpretation:  The model interpretation, provided by the
                    Population Service to which the Experiment Host is connected
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :param model_weights: List of weight tensors of the model, used for
                              weight persistence.
        :return: a dictionary whose keys impart measurements as to the
                 performance of the model.

                 While it is possible for any measurement to be considered
                 the fitness through configuration, by default with no extra
                 configuration, the system looks for a key here called 'fitness'
                 whose value is the primary fitness value.
        """
        raise NotImplementedError
