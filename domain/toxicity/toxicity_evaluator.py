
import os
import pickle
import time
from copy import deepcopy

from future.builtins import range
from past.builtins import basestring

import numpy as np

from keras.layers import Concatenate
from keras.layers import Conv1D
from keras.layers import Embedding
from keras.layers import Input
from keras.layers import SpatialDropout1D
from keras.layers.noise import GaussianNoise
from keras.models import model_from_json
from keras.models import Model
from keras.utils import to_categorical
from keras_preprocessing.sequence import pad_sequences

from sklearn import metrics as sklearn_metrics

from framework.evaluator.keras_network_evaluator import KerasNetworkEvaluator
from framework.evaluator.data_pathdict import get_data_dict_filename
from framework.evaluator.data_pathdict import open_data_dict_file
from framework.soft_ordering.enn.enn_soft_order_multi_task_evaluation \
    import EnnSoftOrderMultiTaskEvaluation


class ToxicityEvaluator(KerasNetworkEvaluator):
    """
    Evaluator class for toxicity problem.

    This gets invoked by reference first by the SessionServer to
    be sure there are no import/syntax errors on the experiment-host
    before handing work off to the Studio Workers.

    The object constructed by the SessionServer is not actually used, however.
    This is for debugging convenience -- it's easier to attach to a local
    session server on the experiment host than on a remote Studio Worker
    performing the evaluation.

    This also gets invoked (and the object heavily used) by the Studio
    Workers actually performing the evaluation.
    """

    def load_data(self, domain_config, data_pathdict):
        """
        Loads and preprocess data for domain

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

        # Do not open embeddings file yet.
        # Do that lazily in apply_embedding_encoder() below.
        # For now, just pass along filenames, but with the same data
        # loading protection errors we would get from loading the file now.
        fasttext = None
        glove = None
        info = domain_config.get('info', {})
        if info.get('evolve_embeddings', False):
            print("Using evolved embeddings")
            fasttext = get_data_dict_filename(data_pathdict, 'fasttext')
            glove = get_data_dict_filename(data_pathdict, 'glove')

        # Create data_dict to return.
        data_dict = {
            'labels': labels,
            'tokens': tokens,
            'fasttext': fasttext,
            'glove': glove
        }

        return data_dict


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

        # Preprocess data based on vocab size and sentence length.
        self.update_vocab_size(global_hyperparameters, domain_config, data_dict)
        self.update_sentence_length(domain_config, data_dict)

        if not isinstance(model_json, basestring):
            # Model is ENNJointSoftModel
            training_model = model_json.training_model
        else:
            # Model is Keras JSON
            core_model = model_from_json(model_json)
            training_model = self.create_training_model(core_model,
                                                  global_hyperparameters,
                                                  domain_config, data_dict)

        if model_weights is not None:
            try:
                # Use set_weights if model_weights is a list of np arrays.
                training_model.set_weights(model_weights)
            except Exception:
                # XXX What exception are we actually looking for here?
                # Use load_weights if model_weights is an h5 file name.
                training_model.load_weights(model_weights)

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

        # Seed for training
        train_seed = self.set_train_seed(candidate_id, domain_config)

        info = domain_config.get('info', {})
        model_evaluation = EnnSoftOrderMultiTaskEvaluation(info)

        # Compile the model
        model_evaluation.compile_model(training_model, global_hyperparameters)

        # Train the model
        metrics = self.train(model_evaluation, training_model,
                                           global_hyperparameters,
                                           domain_config, data_dict)

        # Test the model
        test_val_only = domain_config.get('test_val_only')
        fitness, task_fitness, metrics = self.test(model_evaluation,
                                                   training_model,
                                                   metrics,
                                                   domain_config, data_dict,
                                                   val=test_val_only)

        metrics['task_fitness'] = task_fitness
        metrics['train_seed'] = train_seed
        metrics['fitness'] = fitness
        return metrics


    # Tied for Public Enemy #5 for too-many-branches
    # pylint: disable=too-many-branches
    # Tied for Public Enemy #7 for too-many-statements
    # pylint: disable=too-many-statements
    def apply_embedding_encoder(self, my_input, embedding_size,
                                global_hyperparameters, domain_config,
                                data_dict):

        info = domain_config.get('info', {})

        # Get embeddings info.
        if info['evolve_embeddings']:
            if global_hyperparameters['embeddings'] == 'none':
                pretrained_embeddings = None
                embeddings_trainable = True
            else:
                pretrained_embeddings = global_hyperparameters['embeddings']
                embeddings_trainable = global_hyperparameters.get(
                                                'embeddings_trainable')
        else:
            pretrained_embeddings = info['pretrained_embeddings']
            embeddings_trainable = info['embeddings_trainable']


        # Create embedding encoder.

        vocab_size = global_hyperparameters['vocab_size']
        if pretrained_embeddings is None:
            embedding_layer = Embedding(vocab_size, embedding_size)

            # Apply encoder.
            encoder_output = embedding_layer(my_input)

        else:

            # Use something in case we have no data_dict
            pretrained_embedding_size = embedding_size
            use_weights = None

            # data_dict can be None in the case of visualization
            # on the Experiment Host.
            if data_dict is not None:

                # Load pre-trained embeddings.
                print("Loading embeddings {}".format(pretrained_embeddings))
                with open_data_dict_file(data_dict, pretrained_embeddings) as my_file:
                    embedding_weights = pickle.load(my_file, encoding='latin1')
                print("Embeddings loaded")

                # Trim embeddings down to vocab size.
                # Note: this assumes embeddings were created in decreasing order of
                # word occurence, as in process_data.py.
                embedding_weights = embedding_weights[:vocab_size]

                # Create pre-trained embedding layer. This assumes embedding_weights
                # is a numpy array with shape:
                # (vocab_size, pretrained_embedding_size).
                pretrained_embedding_size = embedding_weights.shape[1]
                use_weights = [embedding_weights]

            embedding_layer = Embedding(
                    vocab_size,
                    pretrained_embedding_size,
                    weights=use_weights,
                    trainable=embeddings_trainable)

            # Create additional raw embeddings if necessary.
            if 'concat_embeddings' in global_hyperparameters:
                concat_embeddings = global_hyperparameters['concat_embeddings']
            else:
                concat_embeddings = info['concat_embeddings']
            if concat_embeddings:
                raw_embedding_layer = Embedding(vocab_size, embedding_size)

            # Create adapter layer to make encoder fit with evolved core model.
            adapter_layer = Conv1D(embedding_size, kernel_size=1)

            # Apply encoder.
            embedding_output = embedding_layer(my_input)
            if concat_embeddings:
                raw_embedding_output = raw_embedding_layer(my_input)
                embedding_output = Concatenate()([embedding_output,
                                                  raw_embedding_output])
            encoder_output = adapter_layer(embedding_output)

        # Apply spatial dropout for regularization.
        if 'input_dropout' in global_hyperparameters:
            input_dropout = global_hyperparameters['input_dropout']
        elif 'input_dropout' in info:
            input_dropout = info['input_dropout']
        else:
            input_dropout = 0.
        if input_dropout > 0.:
            regularization_layer = SpatialDropout1D(input_dropout)
            encoder_output = regularization_layer(encoder_output)

        # Apply spatial dropout for regularization.
        if 'input_dropout' in global_hyperparameters:
            input_dropout = global_hyperparameters['input_dropout']
        elif 'input_dropout' in info:
            input_dropout = info['input_dropout']
        else:
            input_dropout = 0.
        if input_dropout > 0.:
            regularization_layer = SpatialDropout1D(input_dropout)
            encoder_output = regularization_layer(encoder_output)

        # Apply Gaussian noise for regularization.
        if 'input_noise' in global_hyperparameters:
            input_noise = global_hyperparameters['input_noise']
        elif 'input_noise' in info:
            input_noise = info['input_noise']
        else:
            input_noise = 0.
        if input_noise > 0.:
            regularization_layer = GaussianNoise(input_noise)
            encoder_output = regularization_layer(encoder_output)

        return encoder_output


    def create_training_model(self, core_model, global_hyperparameters,
                           domain_config, data_dict):
        """
        :param core_model: ???
        :param global_hyperparameters: These are the
                evolved hyperparameters specific to the candidate, but applied
                globally to the evaluation.  These are specified in the builder
                config by JSON string of evolved data (see README-specs.md).
                If this is not specified, the default contents of this
                dictionary is a single evolved 'learning_rate' double.
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :return: The training model
        """

        info = domain_config.get('info', {})

        num_core_model_inputs = len(core_model.inputs)
        assert num_core_model_inputs in [1, 2], \
                                        "May or may not have constant_input"

        # Set embedding size to be number of filters the core model expects.
        if num_core_model_inputs == 1:
            embedding_size = core_model.input_shape[-1]
        elif num_core_model_inputs == 2:
            embedding_size = core_model.input_shape[-1][-1]

        # Create input layer.
        input_shape = (info['max_sentence_length'],)
        my_input = Input(shape=input_shape)

        # Add encoder.
        encoder_output = self.apply_embedding_encoder(my_input, embedding_size,
                                                      global_hyperparameters,
                                                      domain_config,
                                                      data_dict)

        # Apply core model, adding constant_input if necessary.
        if num_core_model_inputs == 1:
            overall_inputs = my_input
            core_model_inputs = encoder_output
        elif num_core_model_inputs == 2:
            constant_input = Input(shape=(1,))
            overall_inputs = [constant_input, my_input]
            core_model_inputs = [constant_input, encoder_output]
        core_model_output = core_model(core_model_inputs)

        # Add decoder if necessary.
        if info['decoder'] == 'concat_pool':
            from domain.toxicity.decoders import apply_concat_pooling_1d_decoder
            overall_output = apply_concat_pooling_1d_decoder(core_model_output,
                                                             num_classes=2)
        else:
            overall_output = core_model_output

        # Finally, instantiate full model.
        training_model = Model(inputs=overall_inputs, outputs=overall_output)

        return training_model


    # Tied for Public Enemy #5 for too-many-locals
    # pylint: disable=too-many-locals
    def train(self, model_evaluation, training_model, global_hyperparameters,
                domain_config, data_dict):
        """
        Called from evaluate_network() above

        :param model_evaluation: an instance of the common evaluation
                policy class
        :param training_model: the Keras model to train
        :param global_hyperparameters: Short for "global chromosome dictionary",
                these are the evolved hyperparameters specific to the candidate,
                but applied globally to the evaluation.  These are specified in
                the builder config by JSON string of evolved data (see
                README-specs.md).

                If this is not specified, the default contents of this
                dictionary is a single evolved 'learning_rate' double.
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :return: a dictionary containing stats from the training.
        """

        train_start_time = time.time()
        train_info = domain_config.get('info', {})

        checkpoint_dir = domain_config.get('checkpoint_dir')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        labels = data_dict['labels']
        tokens = data_dict['tokens']

        # Set up model checkpointing.
        weights_dir = checkpoint_dir
        best_weights_file_name = os.path.join(weights_dir, 'toxicity_best.hdf5')

        # Add constant_input data if necessary.
        train_inputs = model_evaluation.determine_test_data_inputs(training_model,
                                                               tokens['train'])
        val_inputs = model_evaluation.determine_test_data_inputs(training_model,
                                                             tokens['dev'])
        train_targets = labels['train']
        val_targets = labels['dev']

        # Train.
        epochs = domain_config.get('num_epochs')
        batch_size = domain_config.get('batch_size')
        num_samples = len(train_targets)
        epoch_training_percentage = global_hyperparameters.get(
                                            'epoch_training_percentage', 1)
        samples_per_epoch = int(num_samples * epoch_training_percentage)
        best_val_acc = -float("inf")
        val_acc = -float("inf")
        best_history = None
        for epoch in range(epochs):
            print('epoch', epoch)
            # Get training data for epoch.
            permutation = np.random.permutation(num_samples)[:samples_per_epoch]
            epoch_train_inputs = [X[permutation] for X in train_inputs]
            epoch_train_targets = train_targets[permutation]

            # Train for one epoch.
            history = training_model.fit(x=epoch_train_inputs,
                                  y=epoch_train_targets,
                                  initial_epoch=epoch,
                                  epochs=epoch+1,
                                  validation_data=(val_inputs, val_targets),
                                  batch_size=batch_size)

            # Update best.
            print(list(history.history.keys()))
            val_acc = history.history['val_acc'][-1]
            if val_acc > best_val_acc:
                print('New Best')
                best_val_acc = val_acc
                best_history = deepcopy(history.history)
                training_model.save_weights(best_weights_file_name)
            print("Val acc: {0}; Best Val Acc: {1}".format(
                    val_acc, best_val_acc))

        # Create results dictionary.
        metrics = {
            'loss_history': best_history,
            'avg_gpu_batch_time': [],

            'training_time': time.time() - train_start_time,
            'num_epochs_trained': epochs,
            'total_num_epochs_trained': epochs
        }

        # Load weights of best model.
        if best_val_acc > val_acc:
            print("Loading best weights...")
            training_model.load_weights(best_weights_file_name)
            print("Best weights loaded.")

        metrics['num_params'] = training_model.count_params()
        if train_info['enable_alt_obj']:
            # I.e., maximize the inverse number of parameters.
            metrics['alt_objective'] = -float(metrics['num_params'])

        return metrics


    def test(self, model_evaluation, training_model,
                metrics, domain_config, data_dict, val=True):
        """
        Called from evaluate_network() above

        :param model_evaluation: an instance of the common evaluation policy
                class
        :param training_model: the trained Keras model to test
        :param metrics: a dictionary containing stats gathered during training
        :param val: If True, evaluate on 'valid' split, otherwise on 'test'
                split
        :return: a tuple of overall accuracy, task accuray, and a metrics
                (stats) dictionary from testing.
        """

        if val:
            split = 'dev'
        else:
            split = 'test'

        train_info = domain_config.get('info', {})

        labels = data_dict['labels']
        tokens = data_dict['tokens']

        # Add constant_input data if necessary.
        test_inputs = model_evaluation.determine_test_data_inputs(training_model,
                                                              tokens[split])

        # Test.
        verbose = domain_config.get('verbose', False)
        fitness_metric = train_info['fitness_metric']
        if fitness_metric == 'accuracy':
            loss, accuracy = training_model.evaluate(test_inputs, labels[split])
            fitness = accuracy

            if verbose:
                print("{} loss: {}".format(split, loss))
                print("{} accuracy: {}".format(split, accuracy))

        elif fitness_metric == 'auroc':
            y_pred = training_model.predict(test_inputs)
            auroc = sklearn_metrics.roc_auc_score(labels[split], y_pred)
            fitness = auroc
            print("{} auroc: {}".format(split, auroc))

        return fitness, [fitness], metrics

    def update_vocab_size(self, global_hyperparameters, domain_config,
                          data_dict):
        """
        If vocab_size is evolved, remove tokens greater than the target
        vocab_size. Note that we cannot simply change them to 0, since
        0 is reserved for padding to indicate sentence boundaries.

        :param global_hyperparameters: dictionary which may contain target
                        vocab size
        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :return: Nothing
        """
        info = domain_config.get('info', {})
        if 'vocab_size' in global_hyperparameters:

            vocab_size = global_hyperparameters.get('vocab_size')

            if data_dict is None:
                # We have no tokens to resample.
                # This is the NetworkVisualizer's code path.
                return

            tokens = data_dict['tokens']

            for split in tokens:
                for i in range(len(tokens[split])):
                    sample = tokens[split][i]
                    new_sample = []
                    for j in range(len(sample)):
                        if sample[j] < vocab_size:
                            new_sample.append(sample[j])
                    tokens[split][i] = new_sample
        else:
            global_hyperparameters['vocab_size'] = info['vocab_size']


    def update_sentence_length(self, domain_config, data_dict):
        """
        Pad or trim all sentences to the correct length. In the future
        this number may be evolved and included in global_hyperparameters.

        :param domain_config: The configuration dictionary for domain evaluation
        :param data_dict: the dictionary containing domain keys for each data
                    set used.
        :return: Nothing
        """

        if data_dict is None:
            # We have no tokens to pad.
            # This is the NetworkVisualizer's code path.
            return

        tokens = data_dict['tokens']
        info = domain_config.get('info', {})
        max_sentence_length = info.get("max_sentence_length")
        for split in tokens:
            tokens[split] = pad_sequences(tokens[split],
                                          maxlen=max_sentence_length,
                                          value=0)
