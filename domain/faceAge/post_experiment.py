#!/usr/bin/env python
"""
Harness for training and evaluating models offline after evolution.
"""

import argparse
import logging
import os
import pickle
import gc
import sys
import json
import time

import numpy as np
import matplotlib.pyplot as plt
from keras import backend as K

from framework.evaluator.data_pathdict import generate_data_pathdict
from framework.persistence.candidate_persistence import CandidatePersistence
from framework.resolver.domain_resolver import DomainResolver
from framework.resolver.evaluator_resolver import EvaluatorResolver
from framework.util.memory_util import check_memory_usage

from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence


def get_model_str_and_gcd(args):
    # Load model json string and global_hyperparameters from chromosome file if provided.
    if args.chromosome_file:
        with open(args.chromosome_file, 'rb') as my_file:
            pickle_string = pickle.load(my_file)
        chromosome_dict = pickle_string

        model_json_string = chromosome_dict['interpretation']['model']
        global_hyperparameters = chromosome_dict['interpretation']['global_hyperparameters']

    # Otherwise, load model from json and create default global_hyperparameters.
    elif args.model_json:
        # Load model json string.
        with open(args.model_json, 'rb') as my_file:
            model_json_string = my_file.read()

        # Use default global hyperparameters.
        global_hyperparameters = {'learning_rate': 0.001}

    elif args.candidate_id:
        if not args.generation:
            sys.exit("Please specify --generation with --candidate_id ")
        logger = logging.getLogger("PostExperiment")
        candidate_persistence = CandidatePersistence(
                    args.experiment_dir,
                    candidate_id=args.candidate_id,
                    generation=args.generation,
                    logger=logger)
        candidate = candidate_persistence.restore()
        interpretation = candidate['interpretation']
        global_hyperparameters = interpretation['global_hyperparameters']
        model_json_string = interpretation['model']

    else:
        sys.exit("Please specify either --chromosome_file or --model_json "
                 "or --candidate_id")

    # Use user-defined initial learning rate, if provided.
    if args.initial_lr:
        global_hyperparameters['learning_rate'] = args.initial_lr

    # Use user-specified optimizer, if provided.
    if args.optimizer:
        global_hyperparameters['optimizer'] = args.optimizer

    # Use user-defined epoch training percentage, if provided.
    if args.epoch_training_percentage:
        global_hyperparameters['epoch_training_percentage'] = args.epoch_training_percentage

    return model_json_string, global_hyperparameters


def create_offline_evaluator(args):
    # Resolve classes to use for training.

    # Instantiate training objects.
    domain_resolver = DomainResolver()
    domain_class = domain_resolver.resolve(args.domain_name)
    domain = domain_class()

    evaluator_resolver = EvaluatorResolver()
    evaluator_class = evaluator_resolver.resolve(args.domain_name,
                                            evaluator_name=args.evaluator_name)
    network_evaluator = evaluator_class()

    domain_config = domain.build_config(args.domain_config)

    domain_config['checkpoint_dir'] = args.experiment_dir
    if args.test_only:
        # Disable training if only testing.
        domain_config['num_epochs'] = 0
    if args.custom_train_seed:
        # Use custom seed, e.g., to replicate training from evolution.
        domain_config['custom_train_seed'] = args.custom_train_seed


    return domain_config, network_evaluator


def visualize_training(history_persistence):

    results = history_persistence.restore()
    if not isinstance(results, list):
        results = [results]

    fitness_mat = np.array([x['fitnesses'] for x in results])
    mean = np.mean(fitness_mat, axis=0)
    std = np.std(fitness_mat, axis=0)

    plt.figure(figsize=(20, 16))
    plt.plot(np.arange(len(mean)), mean, color='blue')
    plt.fill_between(np.arange(len(mean)), mean - std/2., mean + std/2.,
        color='blue', alpha=0.3)
    plt.xlabel("Number of Epoches")
    plt.ylabel("Fitness")
    plt.grid()

    history_file = history_persistence.get_file_reference()
    results_file = history_file.replace(".json", "") + ".png"
    plt.savefig(results_file, bbox_inches="tight")
    plt.clf()


def record_results(metrics, experiment_dir, args):

    timestamp = int(time.time())
    base_name = "results_test-only-{0}_time-{1}".format(args.test_only,
                                                        timestamp)
    logger = logging.getLogger("PostExperiment")
    history_persistence = EasyJsonPersistence(base_name=base_name,
                                    folder=experiment_dir,
                                    logger=logger)
    history_persistence.persist(metrics)

    print("Overall results:")
    print(metrics)

    if args.visualize_results:
        print("Visualizing Results")
        visualize_training(history_persistence)


def cleanup(model):
    K.clear_session()
    del model
    gc.collect()


# XXX: These arguments could be made parameters.
def sample_lr(lower_exp=-6, upper_exp=-2):
    return 10**(-(np.random.random() * (upper_exp - lower_exp) - upper_exp))

def _finditem(obj, key, func):
    if not isinstance(obj, dict):
        return
    if key in obj:
        obj[key] = func(obj[key])
    for value in list(obj.values()):
        if isinstance(value, dict):
            _finditem(value, key, func)
        elif isinstance(value, list):
            for element in value:
                _finditem(element, key, func)


def modify_json_string(model_json_string, args):
    if not args.widen_model:
        return model_json_string

    print("Widened model by factor of {}".format(args.widen_model))
    model_dict = json.loads(model_json_string)

    def widen_filter(value):
        return args.widen_model * value
    _finditem(model_dict, "filters", widen_filter)

    def widen_input_shape(value):
        if len(value) != 4:
            return value
        if value == [None, 224, 224, 3]:
            return value
        value[-1] *= args.widen_model
        return value
    _finditem(model_dict, "batch_input_shape", widen_input_shape)

    def widen_target_shape(value):
        if value == [224, 224, 3, 1]:
            return value
        if len(value) == 4:
            value[-2] *= args.widen_model
        if len(value) == 3:
            value[-1] *= args.widen_model
        return value
    _finditem(model_dict, "target_shape", widen_target_shape)

    def widen_repeat_vector(value):
        return args.widen_model * value
    _finditem(model_dict, "n", widen_repeat_vector)

    def widen_padding(value):
        if value == [[0, 0], [0, 0], [0, 0]]:
            return value
        if isinstance(value, list) and len(value) == 3:
            value[-1][-1] = (value[-1][-1] + 3) * args.widen_model - 3
        return value
    _finditem(model_dict, "padding", widen_padding)


    new_model_json_string = json.dumps(model_dict)
    # with open("widened_model.json", "wb") as f:
    #     json.dump(model_dict, f)

    return new_model_json_string


def get_data_pathdict(domain_config, file_dict):

    # Check to see if data file paths are provided, if it is necessary
    # data is loaded using using generate_data_pathdict()
    data_pathdict = None
    if not domain_config.get('dummy_load', False):
        data_pathdict = file_dict
        if file_dict is None:
            data_pathdict = generate_data_pathdict(domain_config,
                                        convert_urls=True)
    return data_pathdict


def evaluate_model_offline(args):
    # Set GPUs to use if applicable.
    if args.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpus
    check_memory_usage()

    # Load model string description and global hyperparameters.
    model_json_string, global_hyperparameters = get_model_str_and_gcd(args)
    model_json_string = modify_json_string(model_json_string, args)

    # Evaluator object to use for training.
    domain_config, network_evaluator = create_offline_evaluator(args)

    # Load data files
    data_pathdict = get_data_pathdict(domain_config, None)

    datadict = None
    if data_pathdict is not None:
        datadict = network_evaluator.load_data(domain_config,
                                               data_pathdict)


    # Initialize file for writing overall results.
    results_summary_file = args.experiment_dir + '_summary.txt'
    with open(results_summary_file, 'w') as my_file:
        pass

    # Evaluate model.
    total_metrics = []
    best_fitness = -float('inf')
    for i in range(args.num_trials):

        # Set random learning rate, if necessary
        if args.random_lr:
            learning_rate = sample_lr()
            global_hyperparameters['learning_rate'] = learning_rate
        else:
            learning_rate = global_hyperparameters['learning_rate']

        # Set checkpoint dir for this trial
        experiment_dir = args.experiment_dir + '_trial_' + str(i) + \
                            '_lr_' + str(learning_rate)
        domain_config['checkpoint_dir'] = experiment_dir


        print("\n\n*****Starting post-experiment trial {}*****\n\n".format(i))

        model, metrics = network_evaluator.evaluate_model(id=None,
                            model=model_json_string,
                            global_hyperparameters=global_hyperparameters,
                            domain_config=domain_config,
                            data_dict=datadict,
                            model_weights=args.model_weights)
        total_metrics.append(metrics)
        cleanup(model)

        # Only keep the model weights if they are the best ever achieved.
        fitness = metrics['fitness']
        if fitness > best_fitness:
            best_fitness = fitness
        else:
            os.remove(experiment_dir + '/toxicity_best.hdf5')

        # Save and display overall training results.

        if len(total_metrics) == 1:
            record_results(total_metrics[0], experiment_dir, args)
        else:
            record_results(total_metrics, experiment_dir, args)

        # Write overall summary to file
        with open(results_summary_file, 'a') as my_file:
            line = '{} {} {}\n'.format(i, fitness, learning_rate)
            my_file.write(line)


def run():
    parser = argparse.ArgumentParser(description="Train and evaluate models after evolution")

    # Model specification arguments. Only one of chromosome_file or
    # model_json needs to be specified.
    parser.add_argument('--chromosome_file', dest='chromosome_file',
                        help='chromosome pickle file to load and train')
    parser.add_argument('--model_json', dest='model_json',
                        help='json file to load and train')
    parser.add_argument('--candidate_id', dest='candidate_id',
                        help='candidate id for json file to load and train')
    parser.add_argument('--generation', dest='generation',
                        help='generation id for json file to load and train')
    parser.add_argument('--model_weights',
                        dest='model_weights',
                        help='Weights file (h5) to load for continuing training or testing.',
                        default=None)

    # Domain specification.
    parser.add_argument('--domain_name',
                        dest='domain_name',
                        help='Domain for evaluating network',
                        default='faceAge')

    # Evaluator specification.
    parser.add_argument('--evaluator_name',
                        dest='evaluator_name',
                        help='NetworkEvaluator implementation to use for training.',
                        default='kerasAdvancedSoftOrderingfaceGender')
    parser.add_argument('--widen_model',
                        dest='widen_model',
                        type=int,
                        help='Increase the depth of the model.',
                        default=None)

    # Training specification.
    parser.add_argument('--num_trials',
                        dest='num_trials',
                        help='Number of times to retrain the network to get statistical results.',
                        type=int,
                        default=1)
    parser.add_argument('--domain_config',
                        dest='domain_config',
                        help='Config file for the domain, including training parameters.',
                        required=True)
    my_help = """Initial learning rate to use instead of default
                 or loading global hyperparameters."""
    parser.add_argument('--initial_lr',
                        dest='initial_lr',
                        type=float,
                        help=my_help,
                        default=None)
    parser.add_argument('--optimizer',
                        dest='optimizer',
                        help='Name of optimizer to use for training. Options: adam, \
                              rmsprop, adadelta, sgd.',
                        default=None)
    parser.add_argument('--epoch_training_percentage',
                        dest='epoch_training_percentage',
                        type=float,
                        help='Use to override optimized or default training amount.',
                        default=None)
    parser.add_argument('--test_only',
                        dest='test_only',
                        help='Use this to test an already trained model.',
                        action='store_true')
    parser.add_argument('--experiment_dir',
                        dest='experiment_dir',
                        help='Directory where experiment results will be saved. \
                             It will be created if it does not already exist. \
                             This should be unique for each offline training run.',
                        required=True)
    parser.add_argument('--custom_train_seed',
                        dest='custom_train_seed',
                        help='Optional seed for training. This may be useful \
                              in replicating results from evolution, i.e., by \
                              using the seed returned \
                              in gen_<gen>/results_dict.json.',
                        type=int,
                        default=None)
    parser.add_argument('--random_lr',
                        dest='random_lr',
                        help='Whether to run each trial with a random \
                              learning rate. This may be useful in doing \
                              final finetuning once the architecture is fixed',
                        action='store_true')

    # Whether to visualize results
    parser.add_argument('--visualize_results',
                        dest='visualize_results',
                        help='Use this to visualize the progress of training.',
                        action='store_true')

    # GPU specification.
    parser.add_argument('-g', '--gpus',
                        dest='gpus',
                        help='Select which GPUs to use, if multiple GPUs are available on the \
                              machine. This is useful so that different users and processes \
                              can use different GPUs. If nothing is specified, all GPU resources \
                              will be allocated. Argument should be a comma-separated list of \
                              the indices of GPUs to use Examples: -g 0; -g 0,1; -g 2; -g 0,1,2,3.',
                        required=False)

    args = parser.parse_args()
    evaluate_model_offline(args)

if __name__ == '__main__':
    run()
