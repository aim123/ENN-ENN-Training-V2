

import datetime
import logging
import os
import random
import sys
import time
import numpy as np

from past.builtins import basestring

from pytz import timezone

from keras.utils import generic_utils
from keras.models import model_from_json
from keras import backend as K

from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence

from framework.evaluator.data_pathdict import get_data_dict_filename
from framework.evaluator.keras_network_evaluator import KerasNetworkEvaluator
from framework.soft_ordering.enn.enn_soft_order_multi_task_evaluation \
    import EnnSoftOrderMultiTaskEvaluation

from domain.omniglot.load_tasks import load_new_dataset


class OmniglotEvaluator(KerasNetworkEvaluator):
    """
    Omniglot model evaluator.
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

        info = domain_config.get('info', {})
        unzip_curr_dir = info.get('unzip_curr_dir', True)

        use_test_data = info.get('use_test_data', False)
        datafile_key = "datafile"
        if use_test_data:
            datafile_key = "omniglot2.tar.gz"

        datafile = get_data_dict_filename(data_pathdict, datafile_key)
        temp_dict = load_new_dataset(datafile,
                                     alphabets=info.get("task_names"),
                                     unzip_curr_dir=unzip_curr_dir)

        data_dict = {
            'X_train': temp_dict['X_train'],
            'Y_train': temp_dict['Y_train'],
            'X_val': temp_dict['X_val'],
            'Y_val': temp_dict['Y_val'],
            'X_test': temp_dict['X_test'],
            'Y_test': temp_dict['Y_test']
        }

        print("Dataset sizes:")
        for name, split_dict in list(data_dict.items()):
            total_sizes = [x.shape for x in list(split_dict.values())]
            print("%s: %s" % (name, total_sizes))

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
            training_model = model_from_json(model_json)

        if model_weights is not None:
            training_model.set_weights(model_weights)

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

        # Seed for training
        train_seed = self.set_train_seed(candidate_id, domain_config)

        info = domain_config.get('info', {})
        model_evaluation = EnnSoftOrderMultiTaskEvaluation(info)

        if not hasattr(training_model, 'optimizer') \
            or training_model.optimizer is None:
            model_evaluation.compile_model(training_model,
                                           global_hyperparameters)

        metrics = self.train(model_evaluation, training_model,
                                           global_hyperparameters,
                                           domain_config, data_dict)
        fitness, task_fitness, metrics = self.test(model_evaluation,
                                                   training_model,
                                                   global_hyperparameters,
                                                   metrics, domain_config,
                                                   data_dict, val=True)
        metrics['task_fitness'] = task_fitness
        metrics['train_seed'] = train_seed
        metrics['fitness'] = fitness
        return metrics


    def test_data_gen(self, val, domain_config, data_dict):
        if val:
            input_test = data_dict['X_val']
            target_test = data_dict['Y_val']
        else:
            input_test = data_dict['X_test']
            target_test = data_dict['Y_test']

        info = domain_config.get('info', {})
        task_names = info.get('task_names')
        for task_name in task_names:
            # print task_name
            yield task_name, input_test[task_name], target_test[task_name]


    def test(self, model_evaluation, training_model, global_hyperparameters,
                metrics, domain_config, data_dict, val=True):

        info = domain_config.get('info', {})
        task_names = info.get('task_names')
        num_tasks = len(task_names)
        generator = self.test_data_gen(val, domain_config, data_dict)

        verbose = domain_config.get('verbose', False)

        task_acc = []
        all_acc = []
        # total = 0.0
        for i, (task_name, input_test, target_test) in enumerate(generator):
            # Note: _ is pythonic for an unused variable. Was: loss
            _, accuracy = model_evaluation.evaluate(i, num_tasks,
                                                       input_test, target_test,
                                                       training_model,
                                                       global_hyperparameters,
                                                       verbose=verbose)
            if verbose:
                print("%s accuracy: %s" % (task_name, accuracy))
            all_acc.append(accuracy)
            task_acc.append((task_name, float(accuracy)))
            # total += 1
            # num_correct += accuracy * input_test.shape[0]
            # total += input_test.shape[0]

        overall_acc = float(np.mean(all_acc))
        if verbose:
            print("Overall test accuracy: %s" % overall_acc)

        return overall_acc, task_acc, metrics

    def augment(self, images, domain_config):

        info = domain_config.get('info', {})
        if not info['augment']:
            return images

        crop_amount = info['crop_amount']
        assert crop_amount % 2 == 0

        img_cols = img_rows = info['image_size'] - crop_amount
        processed_images = []

        for image in images:
            upper = random.randrange(crop_amount + 1)
            lower = upper + img_rows
            left = random.randrange(crop_amount + 1)
            right = left + img_cols
            cropped_image = image[:, upper:lower, left:right, :]
            pad_amount = crop_amount / 2

            padded_image = np.empty((1, info['image_size'],
                                     info['image_size'], 1),
                                    dtype=np.float32)
            padded_image[0, :, :, 0] = np.pad(cropped_image[0, :, :, 0],
                                              pad_amount,
                                              mode="constant")
            processed_images.append(padded_image)
        return processed_images


    def train_data_gen(self, domain_config, data_dict):

        info = domain_config.get('info', {})
        task_names = info.get('task_names')

        input_train = data_dict['X_train']
        target_train = data_dict['Y_train']

        # Note: _ is pythonic for an unused variable. Was: i
        for _ in range(domain_config.get('num_iterations_per_epoch')):
            input_batches = []
            target_batches = []
            for task_name in task_names:
                idxs = np.random.randint(input_train[task_name].shape[0], size=1)
                input_batch = input_train[task_name][idxs]
                target_batch = target_train[task_name][idxs]
                input_batches.append(input_batch)
                target_batches.append(target_batch)

            yield self.augment(input_batches, domain_config), target_batches


    # Public Enemy #2 for too-many-statements
    # pylint: disable=too-many-statements
    # Public Enemy #3 for too-many-branches
    # pylint: disable=too-many-branches
    # Public Enemy #3 for too-many-locals
    # pylint: disable=too-many-locals
    def train(self, model_evaluation, training_model, global_hyperparameters,
               domain_config, data_dict):
        """
        Do training here. After training, generates a trained model and
        saves it to file.
        Gets called by evaluate() below.
        """

        train_start_time = time.time()
        train_config = domain_config
        train_info = domain_config.get('info', {})

        # This starts lower than any possible fitness.
        epoch = 0
        epoch_fitness = best_fitness = -sys.maxsize

        # Initialize results dictionary.
        metrics = {}
        metrics['loss_history'] = []
        metrics['avg_gpu_batch_time'] = []
        if train_config.get('test_every_epoch'):
            metrics['fitnesses'] = []
            if not train_config.get('test_val_only'):
                metrics['test_fitnesses'] = []

        # If checkpoint directory does not not exist, we create it

        abs_checkpoint_dir = os.path.abspath(os.path.expanduser( \
            train_config.get('checkpoint_dir')))
        if train_config.get('checkpoint_dir') != "." \
                and train_config.get('timestamp_chkpt_dir'):
            abs_checkpoint_dir = "%s_%s" % (abs_checkpoint_dir, time.time())
        if not os.path.exists(abs_checkpoint_dir):
            # print abs_checkpoint_dir
            os.makedirs(abs_checkpoint_dir)

        # File for saving the current weights of the trained model.
        id_base = str(global_hyperparameters['id'])
        id_path = os.path.join(abs_checkpoint_dir, id_base)
        model_file = id_path + ".h5"

        # File for saving the weights at the epoch with the best test results.
        best_chromo_file = None

        # File for recording the test results by epoch.
        logger = logging.getLogger("OmniglotEvaluator")
        metrics_persistence = EasyJsonPersistence(base_name=id_base,
                                folder=abs_checkpoint_dir,
                                logger=logger)
        metrics_file = metrics_persistence.get_file_reference()

        # Load checkpoint for this chromosome if it has already been partially
        # trained.
        verbose = train_config.get('verbose', False)
        if train_config.get('load_checkpoint') and \
            os.path.exists(model_file):
            if verbose:
                print("loading weights from file %s" % model_file)
            training_model.load_weights(model_file)

        if train_config.get('load_checkpoint'):
            if verbose:
                print("Attempting to load metrics from file %s" % metrics_file)
            metrics = metrics_persistence.restore()
            if metrics is None:
                metrics = {}
            else:
                epoch = len(metrics['loss_history'])
                best_fitness = self.get_best_fitness(metrics)

        if train_config.get('num_epochs') == 0:
            metrics['training_time'] = 0
            metrics['num_epochs_trained'] = 0
            metrics['total_num_epochs_trained'] = epoch
            return metrics

        # Train loop that does 3 main things:
        # 1. Generates train data into batches
        # 2. Loops through train batches and calls model.train_on_batch
        # 3. At the end of the epoch, calls test
        starting_epoch = epoch
        while True:
            print()
            if verbose:
                print("[%s] Training Epoch %d" % \
                      (datetime.datetime.now(timezone('US/Pacific')) \
                       .strftime("%Y-%m-%d %H:%M:%S %Z%z"), (epoch + 1)))

            batch_size = train_config.get('batch_size')
            progbar = generic_utils.Progbar(
                batch_size * train_config.get('num_iterations_per_epoch'))
            epoch_loss = []
            generator = self.train_data_gen(domain_config, data_dict)
            learning_rate = self.lr_func(train_info, epoch, verbose,
                                   global_hyperparameters)
            K.set_value(training_model.optimizer.lr, learning_rate)

            gpu_batch_times = []
            for input_batch, target_batch in generator:
                gpu_start_time = time.time()

                losses = model_evaluation.train_on_batch(input_batch, target_batch,
                                                         training_model,
                                                         global_hyperparameters)
                mean_loss = np.mean(losses[-train_info["num_tasks"]:])
                gpu_batch_times.append(time.time() - gpu_start_time)
                epoch_loss.append(mean_loss)
                if verbose:
                    progbar.add(batch_size, values=[("train_loss", mean_loss)])
            print()

            epoch += 1

            # Python 3 JSON serialization of metrics does not like numpy floats.
            # Always convert to regular python float before stuffing something
            # into metrics.
            mean_gpu_batch_time = float(round(np.mean(gpu_batch_times), 8))
            metrics['avg_gpu_batch_time'].append((epoch, mean_gpu_batch_time))

            metrics['training_time'] = time.time() - train_start_time
            metrics['num_epochs_trained'] = epoch - starting_epoch
            metrics['total_num_epochs_trained'] = epoch

            # Python 3 JSON serialization of metrics does not like numpy floats.
            # Always convert to regular python float before stuffing something
            # into metrics.
            mean_loss = float(round(np.mean(epoch_loss), 4))
            loss_history_entry = (epoch, mean_loss)
            metrics['loss_history'].append(loss_history_entry)
            if train_config.get('test_every_epoch'):

                # Note: _ is pythonic for an unused variable
                #       Was: epoch_task_fitness
                epoch_fitness, _, metrics = self.test(model_evaluation,
                                training_model,
                                global_hyperparameters, metrics,
                                domain_config, data_dict, val=True)
                metrics['fitnesses'].append((epoch, epoch_fitness))

                if not domain_config.get('test_val_only'):

                    # Note: _ is pythonic for an unused variable
                    #       Was: test_epoch_task_fitness
                    test_epoch_fitness, _, metrics = self.test(model_evaluation,
                                    training_model,
                                    global_hyperparameters, metrics,
                                    domain_config, data_dict, val=False)
                    metrics['test_fitnesses'].append((epoch, test_epoch_fitness))

            # Checkpointing consists of saving network weights and results.
            if train_config.get('checkpoint_interval') is not None \
                and train_config.get('checkpoint_interval') > 0 and \
                    epoch % train_config.get('checkpoint_interval') == 0:
                if verbose:
                    print("saving weights and metrics to file")
                training_model.save_weights(model_file)
                metrics_persistence.persist(metrics)

                # Save network weights as best if they're the best encountered
                # so far.
                if best_fitness < epoch_fitness:
                    best_fitness = epoch_fitness
                    # Remove old best file.
                    if best_chromo_file is not None \
                        and os.path.exists(best_chromo_file):
                        os.remove(best_chromo_file)
                    print("New best fitness achieved. Saving weights...")
                    base = "{0}_E{1}_F{2}_best.h5".format(
                                    global_hyperparameters['id'],
                                    epoch,
                                    round(epoch_fitness, 4))
                    best_chromo_file = os.path.join(abs_checkpoint_dir, base)
                    training_model.save_weights(best_chromo_file)
                    print("Weights saved to {0}".format(best_chromo_file))

            term_criterion = self.termination_criterion(train_config,
                                        train_start_time, starting_epoch,
                                        epoch, metrics['loss_history'])
            if term_criterion != "CONTINUE":
                metrics['term_criterion'] = term_criterion
                break

        if domain_config.get('upload_best_model') and \
            best_chromo_file is not None:
            assert os.path.exists(best_chromo_file)
            metrics['_s3_best_model'] = best_chromo_file
        if domain_config.get('upload_trained_model') and \
            os.path.exists(model_file):
            metrics['_s3_trained_model'] = model_file
        if os.path.exists(metrics_file):
            metrics['_s3_metrics'] = metrics_file

        return metrics


    def get_best_fitness(self, metrics):
        """
        Return the best fitness for any epoch in the results dictionary.
        :param metrics: A metrics dictionary
        :return: the best metrics value
        """
        fitnesses = metrics['fitnesses']
        best_fitness = max([fitness for (epoch, fitness) in fitnesses])
        return best_fitness


    def lr_func(self, train_info, epoch, verbose, global_hyperparameters=None):
        """
        :return: a learning rate
        """

        if global_hyperparameters is None:
            global_hyperparameters = {}

        learning_rate = global_hyperparameters.get('learning_rate', None)
        if learning_rate is None:
            learning_rate = global_hyperparameters.get('lr', None)
        if learning_rate is None:
            learning_rate = train_info['lr']
        learning_rate *= train_info['lr_scale']
        floor = train_info['lr_floor'] * learning_rate if 'lr_floor' in train_info else 0

        new_lr = train_info['lr_decay_amount'] ** \
             (epoch / train_info['lr_decay']) * learning_rate
        if new_lr < floor:
            new_lr = floor

        if verbose:
            print("current learning rate: {}".format(new_lr))

        return new_lr

    def termination_criterion(self, train_config, train_start_time,
                              starting_epoch, epoch, loss_history):

        num_epochs = train_config.get('num_epochs')
        if train_config.get('train_abs_num_epochs'):
            if epoch >= num_epochs:
                return "TERM_EPOCH"
        else:
            if epoch >= starting_epoch + num_epochs:
                return "TERM_EPOCH"

        stop_cond = train_config.get('stop_cond', {})
        timeout_seconds = stop_cond.get('timeout_seconds')
        if time.time() - train_start_time > timeout_seconds:
            return "TERM_TIMEOUT"

        simple_loss_history = [x[1] for x in loss_history]
        farthest_back = stop_cond.get('min_improve_hist_length')

        if len(simple_loss_history) >= farthest_back and \
                simple_loss_history[-farthest_back] - simple_loss_history[-1] \
                < stop_cond['min_improvement']:
            return "TERM_IMPROVEMENT"

        return "CONTINUE"
