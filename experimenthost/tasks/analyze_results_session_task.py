
from __future__ import print_function

import glob

from experimenthost.persistence.best_fitness_candidate_persistence \
        import BestFitnessCandidatePersistence
from experimenthost.persistence.results_dict_persistence \
        import ResultsDictPersistence

from experimenthost.tasks.session_task import SessionTask

from experimenthost.util.candidate_util import CandidateUtil

from experimenthost.networkvisualization.network_multi_visualizer \
    import NetworkMultiVisualizer

from framework.util.experiment_filer import ExperimentFiler
from framework.util.generation_filer import GenerationFiler


class AnalyzeResultsSessionTask(SessionTask):
    """
    SessionTask that performs the AnalyzeResults task.

    This task doesn't actually use the Session object that talks
    to the server, but instead takes all the results files created
    by a run and does some analysis on them.

    XXX What?
    """

    def __init__(self, session, master_config, experiment_dir,
                 fitness_objectives, checkpoint_id=None):
        """
        Constructor.

        :param session: The session with which the task can communicate
                    with the service
        :param master_config: The master config for the task
        :param experiment_dir: The experiment directory for results
        :param fitness_objectives: The FitnessObjectives object
        :param checkpoint_id: The checkpoint id (if any) relevant to the task.
        """
        super(AnalyzeResultsSessionTask, self).__init__(session,
            master_config, experiment_dir, fitness_objectives, checkpoint_id)

        self.candidate_util = CandidateUtil(fitness_objectives)


    def run(self):
        """
        Entry point for the session task execution to take over.
        """

        print("Running AnalyzeResultsSessionTask")

        # Read the results files for each generation.
        # These are written out by write_results_file()

        filer = ExperimentFiler(self.experiment_dir)
        glob_spec = filer.experiment_file("gen_*/results_dict.json")
        results_dicts = glob.glob(glob_spec)

        worker_results_files = sorted(results_dicts)
        if len(worker_results_files) <= 0:
            raise ValueError("No results_dicts.json files found in {0}".format(
                                self.experiment_dir))

        # No generation number needed, we are only looking to
        # parse path components with it.
        generation_filer = GenerationFiler(self.experiment_dir)

        worker_results_dict = {}
        for worker_results_file in worker_results_files:

            generation = generation_filer.get_generation_from_path(
                                                worker_results_file)

            # This slurps in results information returned by workers from all
            # candidates of a specific generation
            results_dict_persistence = ResultsDictPersistence(self.experiment_dir,
                                                          generation,
                                                          logger=self.logger)
            one_worker_results_dict = results_dict_persistence.restore()

            # results_dict here will have one entry per candidate over all
            # generations
            worker_results_dict.update(one_worker_results_dict)

        fitness_objective = self.fitness_objectives.get_fitness_objectives(0)
        is_maximize = fitness_objective.is_maximize_fitness()
        best_result = sorted(list(worker_results_dict.items()),
                            key=lambda \
                            x: max(self.candidate_util.get_candidate_fitness(x)),
                            reverse=is_maximize)[0]
        best_id = best_result.get('id')

        # Open the file of the best candidate.
        best_candidate_persistence = BestFitnessCandidatePersistence(
                                                self.experiment_dir, best_id,
                                                logger=self.logger)
        best_candidate = best_candidate_persistence.restore()

        best_id = self.candidate_util.get_candidate_id(best_candidate)

        self.draw_best_candidate_results(best_candidate,
                                       generation,
                                       suffix='abs')


    def draw_best_candidate_results(self, best_candidate,
                                  generation=None,
                                  suffix=''):

        """
        :param best_candidate: A candidate object comprising the best of a
                        generation.
        :param generation: Default value is None
        :param suffix: Default value is an empty string
        """
        experiment_config = self.master_config.get('experiment_config')
        if not experiment_config.get('visualize'):
            return

        best_id = self.candidate_util.get_candidate_id(best_candidate)
        best_fitness = self.candidate_util.get_candidate_fitness(best_candidate)

        fitness = best_fitness if best_fitness is None else \
            round(best_fitness, 4)

        # Determine the output file name basis

        # XXX Use fitness for now.
        #     Later on can address multi-objective goals.
        metric_name = "fitness"
        if generation is not None:
            # Put the file in the gen_NN directory.
            # Call it best_candidate to match the best_candidate.json
            # that gets put there

            base_name = "best_{0}_candidate".format(metric_name)
            filer = GenerationFiler(self.experiment_dir, generation)
            base_path = filer.get_generation_file(base_name)
        else:
            # We do not have a generation that we know about so write out
            # the old-school file name.
            # XXX Not entirely sure when this path would be taken
            base_name = "F{0}_ID-{1}_{2}best_{3}".format(fitness, best_id,
                                                        suffix, metric_name)
            filer = ExperimentFiler(self.experiment_dir)
            base_path = filer.experiment_file(base_name)

        # NetworkVisualizers use the build_training_model() which requires
        # a data_dict of file keys -> file paths to exist.  Domains that
        # wish to visualize their networks that use the data_dict will
        # need to deal with a None value for data dict in the visualization
        # case.
        data_dict = None

        visualizer = NetworkMultiVisualizer(self.master_config, data_dict,
                                            base_path, logger=self.logger)
        visualizer.visualize(best_candidate)
