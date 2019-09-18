

import math
import multiprocessing
import os
import pickle
import shutil
import time

from past.builtins import basestring

import numpy as np

from keras import backend as K
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from keras.layers import add
from keras.layers import Conv2D
from keras.layers import Input
from keras.layers import MaxPooling2D
from keras.layers import UpSampling2D
from keras.models import model_from_json
from keras.models import Model

from sklearn.metrics import roc_auc_score


from framework.evaluator.data_pathdict import get_data_dict_filename
from framework.evaluator.data_pathdict import open_data_dict_file
from framework.evaluator.keras_network_evaluator import KerasNetworkEvaluator
from framework.soft_ordering.enn.enn_soft_order_multi_task_evaluation \
    import EnnSoftOrderMultiTaskEvaluation

# Custom imports
from domain.faceAge.auroc_history import AurocHistory
from domain.faceAge.data_gen_sequence import DataGenSequence
from domain.faceAge.faceAge_subsampler import stratified_subsample
from domain.faceAge.keras_densenet import dense_net_121


# Number of workers for loading the data
NUM_WORKERS = min(8, multiprocessing.cpu_count())
# Maximum size of queue
MAX_QUEUE_SIZE = min(10, NUM_WORKERS)

#$$$$$$$$$$$$$$$ change start $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
#change the number of image count as per your custom image dataset

NUM_IMAGES = [23708, 0]

#$$$$$$$$$$$$$$$ change end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

class faceAgeEvaluator(KerasNetworkEvaluator):
    """
    Evaluator class for faceGender dataset

    This gets invoked by reference first by the SessionServer to
    be sure there are no import/syntax errors on the experiment-host
    before handing work off to the Studio Workers.
    The object constructed by the SessionServer is not actually used,
    however.

    This is for debugging convenience -- it's easier to attach to a local
    session server on the experiment host than on a remote Studio Worker
    performing the evaluation.

    This also gets invoked (and the object heavily used) by the Studio Workers
    actually performing the evaluation.
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
        info = domain_config.get('info', {})
        unzip_curr_dir = info.get('unzip_curr_dir')
        
        #$$$$$$$$$$$$$$$ change start $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        #modified code for loading images directly from hocon path
        
        extract = True

        if extract:
            datafile = get_data_dict_filename(data_pathdict, 'datafile')
            tar_command = "tar -xzf {0}".format(datafile)
            os.system(tar_command)
            
        fpath = os.path.dirname(os.path.abspath(datafile))
        fpath_flist = os.listdir(fpath)
        for i in fpath_flist:
          isDirectory = os.path.isdir(fpath + '/' +i)
          if(isDirectory):
            image_dir = fpath + '/' +i+'/'

        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("image_dir",image_dir)
        
        #$$$$$$$$$$$$$$$ change end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
          
        # Wait a bit to allow extracted images to flush to disk
        counter = 0
        while len(os.listdir(image_dir)) not in NUM_IMAGES:
            if counter == 5:
                break
            counter += 1
            time.sleep(1.0)
            
        assert len(os.listdir(image_dir)) in NUM_IMAGES
        
      
        
        ###
        with open_data_dict_file(data_pathdict, 'labels') as my_file:
            labels = pickle.load(my_file)
        with open_data_dict_file(data_pathdict, 'partition') as my_file:
            partition = pickle.load(my_file)

        info = domain_config.get('info', {})
        if domain_config.get('subsample'):
            train_tau = info.get('train_subsample_tau', None)
            partition['train'] = stratified_subsample(partition['train'],
                labels, domain_config.get('subsample_amount'), train_tau)
        if domain_config.get('test_subsample'):
            test_subsample_amount = domain_config.get('test_subsample_amount')
            test_tau = info.get('test_subsample_tau', None)
            partition['test'] = stratified_subsample(partition['test'],
                labels, test_subsample_amount, test_tau)
            partition['valid'] = stratified_subsample(partition['valid'],
                labels, test_subsample_amount, test_tau)

        data_dict = {}
        data_dict['image_dir'] = image_dir
        data_dict['labels'] = labels
        data_dict['partition'] = partition
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

        if not isinstance(model_json, basestring):
            # Model is ENNJointSoftModel
            training_model = model_json.training_model
        else:
            # Model is Keras JSON
            core_model = model_from_json(model_json)
            
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("model_json",model_json)
            
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("global_hyperparameters",global_hyperparameters)
            
            print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
            print("domain_config",domain_config)
            
            training_model = self.create_training_model(global_hyperparameters,
                                                  core_model, domain_config)

        if model_weights is not None:
            try:
                # Use set_weights if model_weights is a list of np arrays.
                training_model.set_weights(model_weights)
            except ValueError:
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

        print("GCD: {}".format(global_hyperparameters))

        # Seed for training
        train_seed = self.set_train_seed(candidate_id, domain_config)

        info = domain_config.get('info', {})
        model_evaluation = EnnSoftOrderMultiTaskEvaluation(info)

        model_evaluation.compile_model(training_model, global_hyperparameters)
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("domain_config",domain_config)
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("data_dict",data_dict.keys())

        # Train the model
        metrics = self.train(model_evaluation, training_model,
                                            domain_config, data_dict)

        # Test the model
        test_val_only = domain_config.get('test_val_only', True)
        fitness, task_fitness, metrics = self.test(model_evaluation,
                                                   training_model, metrics,
                                                   domain_config, data_dict,
                                                   val=test_val_only)

        metrics['task_fitness'] = task_fitness
        metrics['train_seed'] = train_seed
        metrics['fitness'] = fitness
        return metrics


    def create_image_encoder(self, global_hyperparameters, input_shape, embedding_size,
                             domain_config):

        info = domain_config.get('info', {})

        # Create encoder.
        if info['encoder'] == 'densenet':
            image_model = dense_net_121(include_top=False,
                                      input_shape=input_shape,
                                      weights=info['encoder_weights'])
        else:
            print("Invalid encoder option {}".format(info['encoder']))

        if isinstance(info['encoder_layer_name'], list):

            layer_outputs = []
            layer_names = []
            for i, layer_name in enumerate(info['encoder_layer_name']):
                layer = image_model.get_layer(layer_name)
                layer_output_shape = layer.output_shape
                assert layer_output_shape[-2] == layer_output_shape[-3]
                layer_output_size = layer_output_shape[-2]

                if "hypercolumn_%s" % i in global_hyperparameters and \
                    not global_hyperparameters["hypercolumn_%s" % i] \
                    and info['encoder_output_size'] \
                    != layer_output_size:
                    continue
                if "larger_kernel_%s" % i in global_hyperparameters and \
                    global_hyperparameters["larger_kernel_%s" % i]:
                    kernel_size = 3
                else:
                    kernel_size = 1

                layer_output = layer.output
                layer_output = Conv2D(embedding_size, padding='same',
                    kernel_size=kernel_size)(layer_output)

                scale_factor = float(layer_output_size) / \
                    info['encoder_output_size']
                if scale_factor > 1:
                    scale = scale_factor
                    assert scale % 2 == 0
                    layer_output = MaxPooling2D(pool_size=(scale, scale))(layer_output)
                elif scale_factor < 1:
                    scale = int(1.0/scale_factor)
                    assert scale % 2 == 0
                    layer_output = UpSampling2D(size=(scale, scale))(layer_output)
                layer_outputs.append(layer_output)
                layer_names.append(layer_name)

            if len(layer_outputs) > 1:
                image_model_output = add(layer_outputs)
            else:
                image_model_output = layer_outputs[0]
            print("Using following hypercolumns from encoder: {}".format(layer_names))

        else:
            layer = image_model.get_layer(info['encoder_layer_name'])
            assert layer.output_shape[-2] == layer.output_shape[-3]
            assert layer.output_shape[-2] == info['encoder_output_size']
            image_model_output = layer.output

            # Create adapter layer to make encoder fit with evolved core model.
            image_model_output = Conv2D(embedding_size, padding='same',
                kernel_size=1)(image_model_output)

            print("Using layer {} from encoder".format(info['encoder_layer_name']))

        image_model_encoder = Model(inputs=image_model.input,
                                    outputs=image_model_output)
        return image_model_encoder


    def create_training_model(self, global_hyperparameters, core_model,
                            domain_config):

        info = domain_config.get('info', {})

        # If necessary, add any auxiliary model components.
        if info['encoder'] is None:
            return core_model

        num_core_model_inputs = len(core_model.inputs)
        assert num_core_model_inputs > 0, \
            "May or may not have constant_input"

        # Set embedding size to be number of filters the core model expects.
        if num_core_model_inputs == 1:
            embedding_size = core_model.input_shape[-1]
        elif num_core_model_inputs >= 2:
            embedding_size = core_model.input_shape[-1][-1]

        # Create input layer.
        input_shape = (info['image_size'],
                       info['image_size'],
                       info['num_channels'])
        input_tensor = Input(shape=input_shape)

        # Add image model encoder.
        image_model_encoder = self.create_image_encoder(global_hyperparameters,
            input_shape, embedding_size, domain_config)
        encoder_output = image_model_encoder(input_tensor)

        # Apply core model, adding constant_input if necessary.
        num_tasks = info['num_tasks']
        constant_input = Input(shape=(1,))
        if not info['multitask']:
            if num_core_model_inputs == num_tasks:
                overall_inputs = input_tensor
                core_model_inputs = encoder_output
            else:
                assert num_core_model_inputs == num_tasks + 1
                overall_inputs = [constant_input, input_tensor]
                core_model_inputs = [constant_input, encoder_output]
        else:
            if num_core_model_inputs == num_tasks:
                overall_inputs = input_tensor
                core_model_inputs = \
                    [encoder_output for i in range(num_tasks)]
            else:
                assert num_core_model_inputs == num_tasks + 1
                overall_inputs = [constant_input, input_tensor]
                core_model_inputs = [constant_input] + \
                    [encoder_output for i in range(num_tasks)]

        core_model_output = core_model(core_model_inputs)

        # Add decoder if necessary.
        # Note: no decoder options currently implemented.
        overall_output = core_model_output

        # Finally, instantiate full model.
        training_model = Model(inputs=overall_inputs, outputs=overall_output)

        return training_model


    # Tied for Public Enemy #10 for too-many-locals
    # pylint: disable=too-many-locals
    # Public Enemy #13 for too-many-branches
    # pylint: disable=too-many-branches
    def test(self, model_evaluation, training_model, metrics,
             domain_config, data_dict, val=True):
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
            split = 'valid'
            current_state = 'validation'
        else:
            split = 'test'
            current_state = 'test'

        train_info = domain_config.get('info', {})

        labels = data_dict['labels']
        partition = data_dict['partition']
        image_dir = data_dict['image_dir']

        # Get test matrix
        
        
        # Instead of len of disease putting num_classes everywhere
        #$$$$$$$$$$$$$$$ change start $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # change range input to train_info['num_classes']
        y_test = np.empty((len(partition[split]) -
                           len(partition[split]) % domain_config.get('batch_size'),
                           train_info['num_classes']), dtype=np.float32)
                           
        #$$$$$$$$$$$$$$$ change end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        
        for i, npy in enumerate(partition[split]):
            if i < len(y_test):
                y_test[i, :] = labels[npy]

        # Evaluate model.
        n_tests = train_info["test_augment"] if train_info["test_augment"] else 1

        y_preds = []
        for i in range(n_tests):
            print("Test augment: {0}, trial: {1}/{2}".format(
                    train_info["test_augment"] is not None, i+1, n_tests))

            y_pred = training_model.predict_generator(
                generator=DataGenSequence(domain_config, train_info, training_model,
                                          model_evaluation, image_dir,
                                          labels, partition[split],
                                          current_state=current_state),
                workers=NUM_WORKERS,
                verbose=domain_config.get('verbose', False),
                max_queue_size=MAX_QUEUE_SIZE)

            # If using a multitask model, predictions must be reformatted.
            if train_info['multitask']:
                new_y_pred = np.empty(y_test.shape, dtype=np.float32)
                
                #$$$$$$$$$$$$$$$ change start $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
                # change range input to train_info['num_classes']
                for index in range(y_test.shape[0]):
                    for j in range(train_info['num_classes']):
                        new_y_pred[index, j] = y_pred[j][index, 1]
                        
                #$$$$$$$$$$$$$$$ change end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

                y_pred = new_y_pred

            y_preds.append(y_pred)

        y_pred = np.mean(y_preds, axis=0) if len(y_preds) > 1 else y_preds[0]
        if np.isfinite(y_pred).all():
            metrics['y_pred_isfinite'] = True
        else:
            metrics['y_pred_isfinite'] = False
            y_pred = np.nan_to_num(y_pred)

        print("Result shape: {}".format(str(y_pred.shape)))
        assert y_pred.shape == y_test.shape

        # Add one fake row of ones in both test and pred values to avoid:
        # ValueError: Only one class present in y_true.
        # ROC AUC score is not defined in that case.
        y_test = np.insert(y_test, 0, np.ones((y_test.shape[1],)), 0)
        y_pred = np.insert(y_pred, 0, np.ones((y_pred.shape[1],)), 0)

        # Calculate average AUROC over all tasks
        task_acc = []
        #$$$$$$$$$$$$$$$ change start $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$
        # change range input to train_info['num_classes']
        for i in range(train_info['num_classes']):
            raw_task_acc = roc_auc_score(y_test[:, i], y_pred[:, i])
            one_task_acc = float(raw_task_acc)
            task_acc.append(one_task_acc)
            
        #$$$$$$$$$$$$$$$ change end $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$    
        
        overall_acc = float(np.mean(task_acc))

        verbose = domain_config.get('verbose', False)
        if verbose:
            print("Task AUROCs are: {}".format(str(task_acc)))
            print("Overall avg test AUROC: {}\n\n\n\n\n\n".format(overall_acc))

        if train_info["dummy_fitness"]:
            overall_acc = math.log(training_model.count_params())

        return overall_acc, task_acc, metrics


    def train(self, model_evaluation, training_model,
              domain_config, data_dict):
        """
        Called from evaluate_network() above

        :param model_evaluation: an instance of the common evaluation policy
                class
        :param training_model: the Keras model to train
        :return: a tuple of the trained model and a dictionary containing stats
                    from the training.
        """
        train_start_time = time.time()

        train_info = domain_config.get('info', {})
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("train_info",train_info)

        checkpoint_dir = domain_config.get('checkpoint_dir')
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        epochs = domain_config.get('num_epochs')

        print("Using at most one GPU")
        labels = data_dict['labels']
        partition = data_dict['partition']
        image_dir = data_dict['image_dir']

        # Set up callbacks.
        weights_dir = checkpoint_dir

        test_every_epoch = domain_config.get('test_every_epoch')
        if test_every_epoch:
            validation_data = DataGenSequence(domain_config, train_info,
                                training_model, model_evaluation, image_dir,
                                labels, partition['valid'],
                                current_state='validation')
            metric = "val_loss"
        else:
            validation_data = None
            metric = "loss"

        model_checkpoint = ModelCheckpoint(
            os.path.join(weights_dir,
                         'chestxray_epoch_{epoch:03d}_%s_{%s:.4f}.hdf5' % (metric, metric)),
            monitor=metric, save_weights_only=True, save_best_only=True)
        initial_lr = float(K.get_value(training_model.optimizer.lr))

        verbose = domain_config.get('verbose', False)
        reduce_lr_on_plateau = ReduceLROnPlateau(monitor=metric, factor=0.1,
                                                 patience=3,
                                                 verbose=verbose,
                                                 min_lr=initial_lr * 0.01)
        callbacks = [model_checkpoint, reduce_lr_on_plateau]

        if test_every_epoch:
            auroc_history = AurocHistory(self, model_evaluation, training_model,
                                            domain_config, data_dict)
            callbacks.append(auroc_history)

        # Train.
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("train_info before training",train_info)
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("domain_config before training",domain_config)
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("image_dir before training",image_dir)
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        #print("labels before training",labels)
        
        print("&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        print("partition['train'] before training",partition['train'])
        
        train_data = DataGenSequence(domain_config, train_info, training_model,
                                     model_evaluation, image_dir,
                                     labels, partition['train'],
                                     current_state='train')
        loss_history = training_model.fit_generator(
            generator=train_data,
            epochs=epochs,
            verbose=verbose,
            callbacks=callbacks,
            workers=NUM_WORKERS,
            max_queue_size=MAX_QUEUE_SIZE,
            validation_data=validation_data
        )

        print("Done Training.")

        # Initialize results dictionary.
        metrics = {}
        metrics['loss_history'] = self.prepare_loss_history(loss_history.history)
        if test_every_epoch:
            metrics['fitnesses'] = auroc_history.aurocs

        metrics['avg_gpu_batch_time'] = []
        metrics['training_time'] = time.time() - train_start_time
        metrics['num_epochs_trained'] = epochs
        metrics['total_num_epochs_trained'] = epochs

        metrics['num_params'] = training_model.count_params()
        if train_info['enable_alt_obj']:
            metrics['alt_objective'] = -1.0 * metrics['num_params']

        return metrics


    def prepare_loss_history(self, loss_history):
        """
        De numpy-ifies the loss history as reported in metrics,
        making this JSON-able.

        XXX Seems we should be able to make this kind of thing generic
            for convenience.

        :param loss_history: The loss history dictionary of layer name to loss list
        :return: Same structure without numpy data types
        """

        if loss_history is None:
            return None

        result = {}
        for key in loss_history.keys():
            loss_list = loss_history.get(key)
            nice_loss_list = []
            for loss in loss_list:
                nice_loss = float(loss)
                nice_loss_list.append(nice_loss)
            result[key] = nice_loss_list

        return result
