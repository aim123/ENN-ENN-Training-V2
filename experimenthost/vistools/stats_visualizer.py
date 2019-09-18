
from __future__ import print_function

import io
from collections import defaultdict

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from experimenthost.networkvisualization.format import FORMAT
from experimenthost.persistence.experiment_host_stats_persistence \
    import ExperimentHostStatsPersistence
from experimenthost.persistence.experiment_host_stats_plot_persistence \
    import ExperimentHostStatsPlotPersistence

matplotlib.use("Agg")


class StatsVisualizer():
    """
    Factored out from the CompletionServiceEvaluatorSessionTask.
    This perhaps might want to be refactored further into a Persistor
    of some kind.  Maybe even two.
    """

    def __init__(self, experiment_dir):
        """
        Constructor

        :param experiment_dir: The directory where experiment results go
        """
        self.experiment_dir = experiment_dir


    def record_and_visualize_stats(self, stats, server_stats):
        """
        Factored out from the CompletionServiceEvaluatorSessionTask.
        """

        print("visualizing stats")
        meant, stdt, medt, trpt, totalt, maxt, meanq, maxq, stdq, succ_rate = \
            stats
        if "serverstats" not in server_stats:
            server_stats["serverstats"] = defaultdict(list)
        my_ss = server_stats["serverstats"]
        my_ss["mean_eval_time"].append(meant)
        my_ss["median_eval_time"].append(medt)
        my_ss["std_eval_time"].append(stdt)
        my_ss["max_eval_time"].append(maxt)
        my_ss["mean_throughput"].append(trpt)
        my_ss["total_eval_time"].append(totalt)
        my_ss["mean_queue_time"].append(meanq)
        my_ss["std_queue_time"].append(stdq)
        my_ss["max_queue_time"].append(maxq)
        my_ss["success_rate"].append(succ_rate)

        # Write out the raw data for the plots in JSON format
        experiment_host_stats_dict = dict(my_ss)
        experiment_host_stats_persistence = ExperimentHostStatsPersistence(
                                        self.experiment_dir)
        experiment_host_stats_persistence.persist(experiment_host_stats_dict)

        xval = np.arange(len(my_ss["mean_eval_time"]))

        def apply_plot_style():
            plt.legend(loc="upper left")
            plt.xlabel("Generation")
            plt.grid()

        fig = plt.figure(figsize=(24, 16))
        plt.subplot(221)
        plt.plot(xval, my_ss["mean_eval_time"], label="mean ind eval time")
        plt.plot(xval, my_ss["std_eval_time"], label="std ind eval time")
        plt.plot(xval, my_ss["max_eval_time"], label="max ind eval time")
        apply_plot_style()

        plt.subplot(222)
        plt.plot(xval, my_ss["mean_throughput"], label="mean throughput")
        ratio = np.divide(my_ss["max_eval_time"], my_ss["total_eval_time"])
        plt.plot(xval, ratio, label="ind/total ratio")
        plt.plot(xval, my_ss["success_rate"], label="success rate")
        apply_plot_style()

        plt.subplot(223)
        plt.plot(xval, my_ss["total_eval_time"], label="total eval time")
        plt.plot(xval, my_ss["max_eval_time"], label="max ind eval time")
        apply_plot_style()

        plt.subplot(224)
        plt.plot(xval, my_ss["mean_queue_time"], label="mean queue wait time")
        plt.plot(xval, my_ss["std_queue_time"], label="std queue wait time")
        plt.plot(xval, my_ss["max_queue_time"], label="max queue wait time")
        apply_plot_style()

        # Save the plot to an image buffer
        plt.tight_layout()
        image_buffer = io.BytesIO()
        plt.savefig(image_buffer, format=FORMAT, bbox_inches="tight")

        # Save the image buffer to a file
        plot_persistence = ExperimentHostStatsPlotPersistence(self.experiment_dir,
                                                          FORMAT)
        plot_persistence.persist(image_buffer)

        plt.close(fig)
