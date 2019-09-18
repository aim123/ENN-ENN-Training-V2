
import os
import time

import pickle
from keras.layers import Embedding
from keras.layers import Input
from keras.models import model_from_json
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from framework.evaluator.data_pathdict import open_data_dict_file
from framework.evaluator.keras_network_evaluator import KerasNetworkEvaluator


class TextClassifierEvaluator(KerasNetworkEvaluator):
    """
    Text classifier model evaluator.

    This gets invoked by reference first by the SessionServer to
    be sure there are no import/syntax errors on the experiment-host
    before handing work off to the Studio Workers.
    The object constructed by the SessionServer is not actually used, however.

    This is for debugging convenience -- it's easier to attach to a local
    session server on the experiment host than on a remote Studio Worker
    performing the evaluation.

    This also gets invoked (and the object heavily used) by the Studio Workers
    actually performing the evaluation.
    """

    def __init__(self):
        """
        Constructor.
        """

        super(TextClassifierEvaluator, self).__init__()

        # Set the vocab_size state variable to a "don't know" value
        # upon class initialization.
        self.vocab_size = None


    def load_data(self, domain_config, data_pathdict):
        """
        Loads and preprocesses the data for the domain.

        :param domain_config: The config dictionary describing the domain
                evaluation parameters
        :param data_pathdict: A dictionary of data files to use
        :return: a single dictionary whose keys describe domain-specific
                    data sets, and whose values are the data sets themselves
                    (often numpy arrays)
        """

        # Load data from file.
        with open_data_dict_file(data_pathdict, 'labels') as my_file:
            labels = pickle.load(my_file)
        with open_data_dict_file(data_pathdict, 'tokens') as my_file:
            tokens = pickle.load(my_file)

        # Convert binary labels to multi-class labels.
        for split in labels:
            labels[split] = to_categorical(labels[split])

        # Process all sentences to the correct length.
        info = domain_config.get('info', {})
        max_sentence_length = info.get("max_sentence_length")
        for split in tokens:
            tokens[split] = pad_sequences(tokens[split],
                    maxlen=max_sentence_length, value=0)

        # Compute vocab size.
        max_index = 0
        for split in tokens:
            for sample in tokens[split]:
                max_index = max(max(sample), max_index)
        self.vocab_size = max_index + 1

        # Subsample if necessary.
        if domain_config.get('subsample'):
            subsample_amount = domain_config.get('subsample_amount')
            labels['train'] = labels['train'][:subsample_amount]
            tokens['train'] = tokens['train'][:subsample_amount]
        if domain_config.get('test_subsample'):
            test_subsample_amount = domain_config.get('test_subsample_amount')
            labels['dev'] = labels['dev'][:test_subsample_amount]
            tokens['dev'] = tokens['dev'][:test_subsample_amount]
            labels['test'] = labels['test'][:test_subsample_amount]
            tokens['test'] = tokens['test'][:test_subsample_amount]

        # Create data_dict to return.
        data_dict = {}
        data_dict['labels'] = labels
        data_dict['tokens'] = tokens

        return data_dict


    def build_training_model(self, candidate_id, model_json,
                        global_hyperparameters, domain_config,
                        data_dict, model_weights=None):
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

        # Create the model.
        core_model = model_from_json(model_json)
        training_model = self.create_training_model(core_model, domain_config)

        # Compile the model
        info = domain_config.get('info', {})
        loss_function = info["loss_function"]
        training_model.compile(optimizer=Adam(lr=global_hyperparameters['learning_rate']),
                            loss=loss_function,
                            metrics=['accuracy'])
                            
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ training_model",training_model.summary())

        return training_model


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

        print("Evaluating {}".format(global_hyperparameters))

        # Set seed for training
        train_seed = self.set_train_seed(candidate_id, domain_config)

        # Train the model
        
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ data_dict",data_dict['labels']['train'][0])
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ domain_config",domain_config)
        training_model, metrics = self.train(training_model, domain_config,
                                             data_dict)

        # Test the model
        fitness, task_fitness, metrics = self.test(training_model, metrics,
                                                   data_dict)

        metrics['task_fitness'] = task_fitness
        metrics['train_seed'] = train_seed
        metrics['fitness'] = fitness
        return metrics


    def create_training_model(self, core_model, domain_config):
        """
        Add word embedding layer at beginning of model.
        """

        # Set embedding size to be number of filters the core model expects.
        embedding_size = core_model.input_shape[-1]

        # Create input layer.
        info = domain_config.get('info', {})
        max_sentence_length = info.get("max_sentence_length")
        input_shape = (max_sentence_length,)
        my_input = Input(shape=input_shape)

        # vocab_size might not have been set in the visualization case.
        if self.vocab_size is None:
            # Get value from the config if it is set there.
            # otherwise use the default from the sample data.
            default_vocab_size = 150000 # XXX?
            self.vocab_size = info.get("vocab_size", default_vocab_size)

        # Add embedding layer.
        embedding_layer = Embedding(self.vocab_size, embedding_size)
        encoder_output = embedding_layer(my_input)

        # Apply core model, adding constant_input if necessary.
        core_model_output = core_model(encoder_output)
        print("$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ core_model_output",core_model_output)
        # Instantiate full model.
        training_model = Model(inputs=my_input, outputs=core_model_output)

        return training_model


    def train(self, training_model, domain_config, data_dict):
        """
        Called from evaluate_network() above

        :param training_model: the Keras model to train
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :return: a tuple of the trained model and a dictionary containing stats
                    from the training.
        """

        train_start_time = time.time()

        checkpoint_dir = domain_config.get('checkpoint_dir')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        labels = data_dict['labels']
        tokens = data_dict['tokens']

        epochs = domain_config.get('num_epochs')
        batch_size = domain_config.get('batch_size')

        # Train.
        history = training_model.fit(x=tokens['train'], y=labels['train'],
                        epochs=epochs,
                        validation_data=(tokens['dev'], labels['dev']),
                        batch_size=batch_size)

        # Create results dictionary.
        metrics = {}
        metrics['loss_history'] = history.history
        metrics['avg_gpu_batch_time'] = []

        metrics['training_time'] = time.time() - train_start_time
        metrics['num_epochs_trained'] = epochs
        metrics['total_num_epochs_trained'] = epochs

        return training_model, metrics


    def test(self, training_model, metrics, data_dict):
        """
        Called from evaluate_network() above

        :param training_model: the trained Keras model to test
        :param metrics: a dictionary containing stats gathered during training
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :return: a tuple of overall accuracy, task accuray, and a metrics
                (stats) dictionary from testing.
        """

        labels = data_dict['labels']
        tokens = data_dict['tokens']

        # Test.
        # Note: _ is pythonic for an unused variable
        _, accuracy = training_model.evaluate(tokens['dev'], labels['dev'])
        fitness = accuracy

        return fitness, [fitness], metrics
