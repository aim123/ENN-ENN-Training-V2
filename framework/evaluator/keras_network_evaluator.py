
import random

import keras
import numpy as np
import tensorflow as tf

from tensorflow import errors

from framework.evaluator.network_evaluator import NetworkEvaluator


class KerasNetworkEvaluator(NetworkEvaluator):
    """
    Base class for neural network evaluation using Keras.
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


    def set_train_seed(self, seed, domain_config):
        """
        :param seed: An integer convertible seed to use
        :param domain_config: a configuration dictionary where the key
                'custom_train_seed' is looked for.
        :return: The seed used
        """

        custom_train_seed = domain_config.get('custom_train_seed', None)
        if custom_train_seed is None:
            train_seed = random.randint(0, 1000000)
        elif custom_train_seed == 'chromo':
            train_seed = int(seed)
        else:
            assert isinstance(custom_train_seed, int)
            train_seed = custom_train_seed

        random.seed(train_seed)
        np.random.seed(train_seed)
        tf.set_random_seed(train_seed)

        return train_seed


    def load_model_weights(self, weights_file):
        """
        :param weights_file: The name of the file with pre-trained weights
        :return: the numpy array containing the weights of the model.
        """
        return keras.models.load_model(weights_file).get_weights()


    def safe_evaluate_network(self, candidate_id, model_json,
                        global_hyperparameters, domain_config, data_dict,
                        model_weights):
        """
        Evaluate the given model as a description of a neural network in a
        matter that allows the network back-end (such as Keras) to look for
        errors specific to the back-end implementation.

        We don't expect specific domains to override this method.
        Instead domains should override load_data() and evaluate_network()
        (see NetworkEvaluator)

        :param candidate_id: the string identifier of the candidate to evaluate
        :param model_json: the Keras JSON string describing the model to train
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

        training_model = None
        metrics = {}
        try:
            # Use the superclass version of this method to do the heavy lifting
            training_model, metrics = \
                super(KerasNetworkEvaluator, self).safe_evaluate_network(
                        candidate_id, model_json, global_hyperparameters,
                        domain_config, data_dict, model_weights)
        except errors.ResourceExhaustedError:
            # XXX This assumes a lot about which metric is measured
            #       for fitness and the range of its values.
            metrics['fitness'] = 0.0

        return training_model, metrics
