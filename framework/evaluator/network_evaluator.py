
import os

import numpy as np

from studio.fs_tracker import get_artifact

from framework.evaluator.model_evaluator import ModelEvaluator


class NetworkEvaluator(ModelEvaluator):
    """
    A partial implementation of a Model Evaluator for Neural Networks
    that has some common code for optional persistence of model weights.

    Implementations really only need to worry about implementing
    load_data() and evaluate_network().
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


    def build_training_model(self, candidate_id, model_json,
                             global_hyperparameters, domain_config, data_dict,
                              model_weights=None):
        """
        Build the training model from a description of a neural network.

        This is separated out from evaluate_network() below
        so common weight persistence logic can be used, if desired.

        :param candidate_id: the string identifier of the candidate to evaluate
        :param model_json: the JSON string describing the "creamy center"
                    of the model to create
        :param global_hyperparameters: These are the
                evolved hyperparameters specific to the candidate, but applied
                globally to the evaluation.  These are specified in the builder
                config by JSON string of evolved data (see README-specs.md).
                If this is not specified, the default contents of this
                dictionary is a single evolved 'learning_rate' double.
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used. Only in the case of calling this method for
                    Network Visualizers will this argument be called
                    with a None value.  Domains that wish to visualize their
                    networks that use the data_dict will need to deal with a
                    None value for data dict in the visualization case.
        :param model_weights: List of weight tensors of the model, used for
                              weight persistence.
        :return: The model to train, with all extra input, output and
                    data augmentation layers attached.
        """
        raise NotImplementedError


    def evaluate_network(self, candidate_id, training_model,
                         global_hyperparameters, domain_config, data_dict):
        """
        Evaluate the given model as a description of a neural network.

        :param candidate_id: the string identifier of the candidate to evaluate
        :param training_model: the Keras model to train and evaluate
        :param global_hyperparameters: These are the
                evolved hyperparameters specific to the candidate, but applied
                globally to the evaluation.  These are specified in the builder
                config by JSON string of evolved data (see README-specs.md).
                If this is not specified, the default contents of this
                dictionary is a single evolved 'learning_rate' double.
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :return: A dictionary whose keys impart measurements as to the
                 performance of the model.

                 While it is possible for any measurement to be considered
                 the fitness through configuration, by default with no extra
                 configuration, the system looks for a key here called 'fitness'
                 whose value is the primary fitness value.
        """
        raise NotImplementedError


    def evaluate_model(self, candidate_id, interpretation, domain_config,
                        data_dict, model_weights=None):
        """
        Evaluate the given model interpretation.
        Fulfills superclass interface.
        Most subclasses should not need to override this method.

        This is the main entry point for candidate evaluation,
        called by the client.py worker entry point script.

        :param id: the string identifier of the candidate to evaluate
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

        verbose = domain_config.get('verbose', False)
        model_weights = self.unpack_weights(verbose)

        model_json = interpretation.get('model', None)
        global_hyperparameters = interpretation.get('global_hyperparameters',
                                                    None)

        trained_model, metrics = self.safe_evaluate_network(candidate_id,
                                    model_json, global_hyperparameters,
                                    domain_config, data_dict, model_weights)

        if verbose and trained_model is not None:
            my_l2norm = self.l2norm(trained_model.get_weights())
            print("weight L2 norm = {}".format(my_l2norm))

        persist_weights = domain_config.get('persist_weights', None)
        self.pack_weights(persist_weights, trained_model, metrics, verbose)

        return metrics


    def l2norm(self, weights):
        norm = 0
        for weight in weights:
            norm += np.linalg.norm(weight)
        return float(norm)


    def get_model_name(self):
        return 'model.h5'


    def load_model_weights(self, weights_file):
        """
        :param weights_file: The name of the weights file
        :return: The numpy array containing model weights
        """
        raise NotImplementedError


    def unpack_weights(self, verbose):
        """
        Load weights if present
        """

        try:
            weights_file = os.path.join(get_artifact('modeldir'),
                                        self.get_model_name())
            if verbose:
                print("Loading weights from {}".format(weights_file))
            weights = self.load_model_weights(weights_file)
            if verbose:
                print("Loaded successfully, L2 norm of weights = {}".format(
                        self.l2norm(weights)))
            return weights
        except BaseException as exception:
            if verbose:
                print("Weight loading failed due to {}".format(exception))
                print("unpack_weights returns None")
            return None


    def pack_weights(self, persist_weights, model, metrics, verbose):
        """
        Save weights if persist_weights flag is True
        """

        if model is None:
            # This happens if a ResourceExhaustedError is caught
            metrics['weights_l2norm'] = None
            return

        if persist_weights:
            metrics['weights_l2norm'] = self.l2norm(model.get_weights())
            weights_file = os.path.join(get_artifact('modeldir'),
                                        self.get_model_name())
            if verbose:
                print("Saving weights file to {}".format(weights_file))
            model.save(weights_file)
            if verbose:
                print("Saving complete")


    def safe_evaluate_network(self, candidate_id, model_json,
                        global_hyperparameters, domain_config, data_dict,
                        model_weights):
        """
        Evaluate the given model as a description of a neural network in a
        matter that allows the network back-end (such as Keras) to look for
        errors specific to the back-end implementation.

        We don't expect specific domains to override this method.

        :param candidate_id: the string identifier of the candidate to evaluate
        :param model_json: the JSON string describing the model to train
                        and evaluate
        :param global_hyperparameters: These are the
                evolved hyperparameters specific to the candidate, but applied
                globally to the evaluation.  These are specified in the builder
                config by JSON string of evolved data (see README-specs.md).
                If this is not specified, the default contents of this
                dictionary is a single evolved 'learning_rate' double.
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :param model_weights: List of weight tensors of the model, used for
                              weight persistence.
        :return: a tuple of:
                    1. The model itself. This is so common weight persistence
                        logic can be used, if desired.
                    2. A dictionary whose keys impart measurements as to the
                       performance of the model.

                 While it is possible for any measurement to be considered
                 the fitness through configuration, by default with no extra
                 configuration, the system looks for a key here called 'fitness'
                 whose value is the primary fitness value.
        """
        # By default, do not look for any errors
        training_model = self.build_training_model(candidate_id,
                            model_json, global_hyperparameters,
                            domain_config, data_dict, model_weights)
        metrics = self.evaluate_network(candidate_id, training_model,
                        global_hyperparameters, domain_config, data_dict)

        return training_model, metrics
