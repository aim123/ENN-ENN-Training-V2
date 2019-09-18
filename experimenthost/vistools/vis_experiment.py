#!/usr/bin/env python
import os
import glob
import argparse

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from keras.models import model_from_json
import keras.backend as K

from experimenthost.persistence.results_dict_persistence \
    import ResultsDictPersistence

# XXX Use a NetworkVisualizer?
from experimenthost.vistools.vis_util import plot_model

from framework.persistence.candidate_persistence import CandidatePersistence
from framework.util.experiment_filer import ExperimentFiler
from framework.util.generation_filer import GenerationFiler

matplotlib.use('TkAgg')


def resolve_alt_objective(args):
    if args.alt_objective == "training_time":
        label = 'Training Time (Hours)'
    elif args.alt_objective == 'num_params':
        label = 'Number of Parameters'
    else:
        label = "Alt Objective"
    return label

def get_results_files(args):

    filer = ExperimentFiler(args.experiment_dir)
    glob_spec = filer.experiment_file("*/results_dict.json")
    glob_results = glob.glob(glob_spec)
    results_files = sorted(glob_results)
    if args.max_gen > 0:
        results_files = results_files[:args.max_gen]
    return results_files

def visualize_training_runs(args):
    total_training_times = []
    longest_training_times = []
    average_training_times = []

    results_files = get_results_files(args)
    num_generations = len(results_files)

    generation_filer = GenerationFiler(args.experiment_dir)
    for results_file in results_files:
        generation = generation_filer.get_generation_from_path(results_file)
        persistence = ResultsDictPersistence(args.experiment_dir, generation,
                                            logger=None)
        results_dict = persistence.restore()

        if len(results_dict) == 0:
            # File not found
            continue

        times = []
        for key in results_dict.keys():
            result = results_dict[key]
            try:
                training_time = result['metrics']['training_time']
            except Exception:
                training_time = result['metrics']['training_time']
            times.append(training_time)
        total_training_times.append(np.sum(times) / 3600)
        longest_training_times.append(np.max(times) / 3600)
        average_training_times.append(np.mean(times) / 3600)

    plt.plot(total_training_times)
    plt.title('Total Training Machine Hours per Generation')
    plt.ylabel('Hours')
    plt.xlabel('Generation')
    plt.xlim(0, num_generations)

    filer = ExperimentFiler(args.experiment_dir)
    runs_total_file = filer.experiment_file("training_runs_total.png")
    plt.savefig(runs_total_file, bbox_inches='tight')
    plt.clf()

    plt.plot(longest_training_times)
    plt.title('Longest Training Machine Hours per Generation')
    plt.ylabel('Hours')
    plt.xlabel('Generation')
    plt.xlim(0, num_generations)

    runs_longest_file = filer.experiment_file("training_runs_longest.png")
    plt.savefig(runs_longest_file, bbox_inches='tight')
    plt.clf()

    plt.plot(average_training_times)
    plt.title('Mean Training Machine Hours per Generation')
    plt.ylabel('Hours')
    plt.xlabel('Generation')
    plt.xlim(0, num_generations)

    runs_avg_file = filer.experiment_file("training_runs_avg.png")
    plt.savefig(runs_avg_file, bbox_inches='tight')
    plt.clf()

    total_machine_hours = np.sum(total_training_times)
    print("Total machine hours used: {}".format(total_machine_hours))


def get_model(experiment_dir, candidate_id, generation):
    candidate_persistence = CandidatePersistence(experiment_dir,
                                             candidate_id=candidate_id,
                                             generation=generation)
    candidate = candidate_persistence.restore()
    interpretation = candidate.get("interpretation", None)
    model_json = interpretation.get("model")

    K.clear_session()
    model = model_from_json(model_json)
    return model

def visualize_model(args):

    model_name = args.visualize_model
    model = get_model(args.experiment_dir, model_name, args.generation)

    filer = ExperimentFiler(args.experiment_dir)
    blueprint_file = filer.experiment_file("{0}_blueprint.png".format(
                                            model_name))

    # XXX Use a NetworkVisualizer
    plot_model(model, blueprint_file, show_layer_names=True,
                draw_submodels_only=True)
    for layer in model.layers:
        try:
            layer_file = filer.experiment_file("{0}_module-{1}.png".format(
                                            model_name, layer.name))
            # XXX Use a NetworkVisualizer
            plot_model(layer, layer_file, show_layer_names=False)
        except Exception:
            print("{} is not a model".format(layer))

def write_training_runs(args):

    filer = ExperimentFiler(args.experiment_dir)
    csv_file = filer.experiment_file("training_runs.csv")
    with open(csv_file, 'wb') as my_file:
        my_file.write('Generation, %s, Fitness\n' % resolve_alt_objective(args))

    results_files = get_results_files(args)

    generation_filer = GenerationFiler(args.experiment_dir)
    for results_file in results_files:
        generation = generation_filer.get_generation_from_path(results_file)
        persistence = ResultsDictPersistence(args.experiment_dir, generation,
                                             logger=None)
        results_dict = persistence.restore()

        if len(results_dict) == 0:
            # File not found
            continue

        for key in results_dict.keys():
            result = results_dict[key]
            try:
                # XXX Use FitnessObjectives prepared from config?
                alt_objective = result['metrics'][args.alt_objective]
                fitness = result['metrics']['fitness'] # XXX not kosher
            except Exception:
                try:
                    alt_objective = result['metrics'][args.alt_objective]
                    fitness = result['fitness'] # XXX not kosher
                except Exception as exception:
                    if args.alt_objective == "num_params":
                        fitness = result['fitness'] # XXX not kosher

                        # XXX What generates this params file?
                        cache_file = generation_filer.get_generation_file(
                                        "candidate_{0}.params".format(key))

                        if os.path.exists(cache_file):
                            with open(cache_file, 'rb') as my_file:
                                alt_objective = my_file.read()
                        else:
                            undefined = 0
                            k = undefined  # XXX
                            print("Extracting num params from network {}".format(k))
                            model = get_model(args.experiment_dir, key, generation)
                            alt_objective = str(model.count_params())
                            with open(cache_file, 'wb') as my_file:
                                my_file.write(alt_objective)
                    else:
                        raise exception

            if args.alt_objective == 'training_time':
                alt_objective = str(float(alt_objective) / 3600.0)
            with open(csv_file, 'ab') as my_file:
                line = '%s %s %s\n' % (generation, alt_objective, fitness)
                my_file.write(line)
    return csv_file

def generate_pareto_front(results_file):
    gens = []
    alt_objs = []
    accs = []
    with open(results_file, 'rb') as my_file:
        for i, line in enumerate(my_file.readlines()):
            if i == 0:
                continue
            data = line.split()
            gens.append(int(data[0]))
            alt_objs.append(float(data[1]))
            accs.append(float(data[2]))

    pareto_gens = []
    pareto_alt_objs = []
    pareto_accs = []
    print("Number results: {} ".format(len(gens)))
    # Note: _ is pythonic for unused variable
    for i, _ in enumerate(gens):
        on_front = True
        acc = accs[i]
        alt_obj = alt_objs[i]
        for j, _ in enumerate(gens):
            if i != j and alt_objs[j] <= alt_obj and accs[j] >= acc:
                on_front = False
                break
        if on_front:
            print("Adding {} {} {}".format(gens[i], alt_obj, acc))
            pareto_gens.append(gens[i])
            pareto_alt_objs.append(alt_obj)
            pareto_accs.append(acc)

    sorted_gens = [gen for alt_obj, gen in sorted(zip(pareto_alt_objs, pareto_gens))]
    sorted_accs = [acc for alt_obj, acc in sorted(zip(pareto_alt_objs, pareto_accs))]
    sorted_alt_objs = sorted(pareto_alt_objs)
    return sorted_gens, sorted_accs, sorted_alt_objs

def visualize_pareto_front(args, sorted_gens, sorted_accs, sorted_alt_objs,
    experiment_names=None):

    plt.figure(figsize=(25, 15))
    plt.xlabel(resolve_alt_objective(args))
    plt.title('Fitness vs. %s Tradeoff' % resolve_alt_objective(args))
    plt.ylabel('Fitness')
    colors = ['g', 'b', 'r', 'm', 'c', 'y'] * 100

    if experiment_names is not None:
        assert len(experiment_names) == len(sorted_gens) == \
            len(sorted_accs) == len(sorted_alt_objs)
        num_experiments = len(experiment_names)
    else:
        num_experiments = 1
        sorted_gens = [sorted_gens]
        sorted_accs = [sorted_accs]
        sorted_alt_objs = [sorted_alt_objs]

    for i in range(num_experiments):
        label = None if experiment_names is None else experiment_names[i]
        plt.scatter(sorted_alt_objs[i], sorted_accs[i], alpha=0.5, marker='.',
            label=label, color=colors[i])
        min_fit = min(sorted_accs[i])
        max_alt = max(sorted_alt_objs[i])
        for j in range(len(sorted_gens[i])):

            # Get relevant values.
            if j == 0:
                prev_fit = min_fit
            else:
                prev_fit = sorted_accs[i][j-1]
            if j == len(sorted_gens[i]) - 1:
                next_alt = max_alt
            else:
                next_alt = sorted_alt_objs[i][j+1]
            curr_fit = sorted_accs[i][j]
            curr_alt = sorted_alt_objs[i][j]

            # Plot vertical line up to point.
            plt.plot([curr_alt, curr_alt], [prev_fit, curr_fit], color=colors[i])

            # Plot horizontal line to next point.
            plt.plot([curr_alt, next_alt], [curr_fit, curr_fit], color=colors[i])

            # Label the new point.

            label_x = 30
            plt.annotate(str(sorted_gens[i][j]),
                (sorted_alt_objs[i][j], sorted_accs[i][j]),
                color='black', xytext=(label_x, -label_x),
                textcoords='offset points',
                arrowprops=dict(arrowstyle="->", color=colors[i], lw=0.5))

    if (args.min_x is not None) and (args.max_x is not None):
        plt.xlim(args.min_x, args.max_x)
    if (args.min_y is not None) and (args.max_y is not None):
        plt.ylim(args.min_y, args.max_y)

    if experiment_names is not None:
        plt.legend()
    # Turn on the minor TICKS, which are required for the minor GRID
    plt.minorticks_on()
    # Customize the major grid
    plt.grid(which='major', linestyle='-', alpha=0.5, linewidth='0.8')
    # Customize the minor grid
    plt.grid(which='minor', linestyle=':', alpha=0.5, linewidth='0.5')

    plt.tight_layout()

    filer = ExperimentFiler(args.experiment_dir)
    runs_pareto_file = filer.experiment_file("training_runs_pareto.png")
    plt.savefig(runs_pareto_file)
    plt.clf()

def compare_pareto_fronts(args):
    gen_list = []
    acc_list = []
    alt_objs_list = []
    exp_names = []
    for experiment_dir in args.compare_pareto:
        args.experiment_dir = experiment_dir
        results_file = write_training_runs(args)
        sorted_gens, sorted_accs, sorted_alt_objs = generate_pareto_front(results_file)
        gen_list.append(sorted_gens)
        acc_list.append(sorted_accs)
        alt_objs_list.append(sorted_alt_objs)
        experiment_name = os.path.basename(os.path.normpath(experiment_dir))
        exp_names.append(experiment_name)
    args.experiment_dir = '.'
    visualize_pareto_front(args, gen_list, acc_list, alt_objs_list, exp_names)

def visualize_main(args):
    if args.visualize_runs:
        assert args.experiment_dir is not None
        visualize_training_runs(args)
        results_file = write_training_runs(args)
        sorted_gens, sorted_accs, sorted_alt_objs = generate_pareto_front(results_file)
        visualize_pareto_front(args, sorted_gens, sorted_accs, sorted_alt_objs)
    if args.visualize_model:
        assert args.experiment_dir is not None
        assert args.generation is not None
        visualize_model(args)
    if args.compare_pareto:
        assert args.compare_pareto is not None
        compare_pareto_fronts(args)

def create_parser():
    parser = argparse.ArgumentParser(
        description="Visualization and summarization of statistics from experiment results")

    parser.add_argument('--experiment_dir', dest='experiment_dir',
                        help='Experiment results directory',
                        default=None)
    parser.add_argument('--visualize_runs',
                        dest='visualize_runs',
                        help=\
                        'Visualize statistics, fitness and secondary objective, for training runs',
                        action='store_true')
    parser.add_argument('--generation',
                        dest='generation',
                        help='Generation number for --visualize_model',
                        default=None)
    parser.add_argument('--visualize_model',
                        dest='visualize_model',
                        help='Visualize blueprint and modules for the provided model file',
                        default=None)
    parser.add_argument('--compare_pareto',
                        dest='compare_pareto',
                        nargs='+',
                        help='Generate a combined pareto plot of all experiments listed',
                        default=None)
    parser.add_argument('--alt_objective',
                        dest='alt_objective',
                        help='Secondary objective to visualize, by default it is training time',
                        default='training_time')
    parser.add_argument('--max_gen',
                        dest='max_gen',
                        help='Maximum number of generations to compute statistics for results',
                        type=int,
                        default=0)
    parser.add_argument('--min_x',
                        dest='min_x',
                        help='Min x-value on pareto plot',
                        type=float,
                        default=None)
    parser.add_argument('--max_x',
                        dest='max_x',
                        help='Max x-value on pareto plot',
                        type=float,
                        default=None)
    parser.add_argument('--min_y',
                        dest='min_y',
                        help='Min y-value on pareto plot',
                        type=float,
                        default=None)
    parser.add_argument('--max_y',
                        dest='max_y',
                        help='Max y-value on pareto plot',
                        type=float,
                        default=None)
    return parser

if __name__ == "__main__":
    PARSER = create_parser()
    MAIN_ARGS = PARSER.parse_args()
    visualize_main(MAIN_ARGS)
