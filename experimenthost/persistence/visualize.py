# -*- coding: UTF-8 -*-

import os
import random
import traceback
import glob

from future.builtins import range

USE_PYPLOT = True
if USE_PYPLOT:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        HAS_PYPLOT = True
    except ImportError:
        print("Matplotlib is not installed.")
        HAS_PYPLOT = False
else:
    try:
        import biggles  # Requires biggles: http://biggles.sourceforge.net/

        HAS_BIGGLES = True
    except ImportError:
        print("Biggles is not installed. If you wish to automatically plot")
        print("some nice statistics please install it:")
        print("http://biggles.sourceforge.net/")
        HAS_BIGGLES = False


def plot_stats(stats, candidate_util, dest_dir='.'):
    """
    Plots the population's average and best fitness.
    :param stats: The stats to plot(?)
    :param candidate_util: A CandidateUtil instance which is already set up
                with the config's notion of which metrics are fitness objectives
    :param dest_dir: Where the visualization should go(?)
    """
    if USE_PYPLOT:
        plot_stats_pyplot(stats, candidate_util, dest_dir)
    else:
        plot_stats_biggles(stats, candidate_util, dest_dir)


def plot_species(species_log, dest_dir='.'):
    """
    Visualizes speciation throughout evolution.
    """
    if USE_PYPLOT:
        plot_species_pyplot(species_log, dest_dir)
    else:
        plot_species_biggles(species_log, dest_dir)


def plot_stats_pyplot(stats, candidate_util, dest_dir='.'):
    if HAS_PYPLOT:
        generation = [i for i in range(len(stats[0]))]

        best_fit = [candidate_util.get_candidate_fitness(c) \
                        if candidate_util.get_candidate_fitness(c) is not None \
                        else 0 for c in stats[0]]
        avg_fit = [avg if avg is not None else 0 for avg in stats[1]]

        fig = plt.figure(figsize=(13, 8))
        subplot = fig.add_subplot(1, 1, 1)
        plt.plot(generation, best_fit, marker='o', label='best fitness')
        plt.plot(generation, avg_fit, marker='x', label='average fitness')

        plt.title("Population's Average and Best Fitness")
        plt.ylabel("Fitness")
        plt.xlabel("Generations")
        subplot.set_xticks(generation, minor=True)
        plt.legend(loc="lower right")
        plt.grid()

        for fitness_graph in glob.glob(os.path.join(dest_dir, "avg_fitness*")):
            os.remove(fitness_graph)
        plt.savefig(os.path.join(dest_dir, 'avg_fitness_g-%s_f-%s.png' %
                                 (generation[-1], round(best_fit[-1], 4))),
                    bbox_inches="tight", dpi=200)

        plt.close()
    else:
        print('You do not have the Matplotlib package.')


def plot_species_pyplot(species_log, dest_dir='.'):
    if HAS_PYPLOT:
        random.seed(0)
        generation = [i for i in range(len(species_log))]
        species = []
        curves = []
        for gen in range(len(generation)):
            for j in range(len(species_log), 0, -1):
                try:
                    species.append(species_log[-j][gen] + sum(species_log[-j][:gen]))
                except IndexError:
                    species.append(sum(species_log[-j][:gen]))
            curves.append(species)
            species = []

        fig = plt.figure(figsize=(13, 8))
        subplot = fig.add_subplot(1, 1, 1)
        plt.fill_between(generation, [0] * len(generation), curves[0],
                         color=(random.random(), random.random(), random.random()))
        for i in range(1, len(curves)):
            plt.fill_between(generation, curves[i - 1], curves[i],
                             color=(random.random(), random.random(), random.random()))

        plt.title("Speciation")
        plt.ylim((0, max(curves[-1])))
        plt.ylabel("Size per Species")
        plt.xlabel("Generations")
        subplot.set_xticks(generation, minor=True)
        plt.savefig(os.path.join(dest_dir, 'speciation.png'),
                    bbox_inches="tight", dpi=200)
        plt.close()

    else:
        print('You do not have the Matplotlib package.')


def plot_stats_biggles(stats, candidate_util, dest_dir='.'):
    if HAS_BIGGLES:
        generation = [i for i in range(len(stats[0]))]

        fitness = [candidate_util.get_candidate_fitness(c) \
                    if candidate_util.get_candidate_fitness(c) is not None \
                    else 0 for c in stats[0]]
        avg_pop = [avg if avg is not None else 0 for avg in stats[1]]

        plot = biggles.FramedPlot()
        plot.title = "Population's average and best fitness"
        plot.xlabel = r"Generations"
        plot.ylabel = r"Fitness"

        plot.add(biggles.Curve(generation, fitness, color="red"))
        plot.add(biggles.Curve(generation, avg_pop, color="blue"))

        # plot.show() # X11
        try:
            plot.write_img(1300, 800, os.path.join(dest_dir, 'avg_fitness.png'))
        except Exception:
            print(traceback.format_exc())
        # width and height doesn't seem to affect the output!
    else:
        print('You do not have the Biggles package.')


def plot_species_biggles(species_log, dest_dir='.'):
    if HAS_BIGGLES:
        plot = biggles.FramedPlot()
        plot.title = "Speciation"
        plot.ylabel = r"Size per Species"
        plot.xlabel = r"Generations"
        generation = [i for i in range(len(species_log))]

        species = []
        curves = []

        for gen in range(len(generation)):
            for j in range(len(species_log), 0, -1):
                try:
                    species.append(species_log[-j][gen] + sum(species_log[-j][:gen]))
                except IndexError:
                    species.append(sum(species_log[-j][:gen]))
            curves.append(species)
            species = []

        new_curve = biggles.Curve(generation, curves[0])
        plot.add(new_curve)
        plot.add(biggles.FillBetween(generation,
                                     [0] * len(generation),
                                     generation,
                                     curves[0],
                                     color=random.randint(0, 90000)))

        for i in range(1, len(curves)):
            curve = biggles.Curve(generation, curves[i])
            plot.add(curve)
            plot.add(biggles.FillBetween(generation,
                                         curves[i - 1],
                                         generation,
                                         curves[i],
                                         color=random.randint(0, 90000)))

        try:
            plot.write_img(1300, 800, os.path.join(dest_dir, 'speciation.png'))
        except Exception:
            print(traceback.format_exc())

    else:
        print('You do not have the Biggles package.')
