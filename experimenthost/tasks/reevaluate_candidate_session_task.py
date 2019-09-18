
from __future__ import print_function

import os

from future.builtins import range

from framework.persistence.candidate_persistence \
    import CandidatePersistence

from experimenthost.persistence.best_fitness_candidate_persistence \
    import BestFitnessCandidatePersistence
from experimenthost.tasks.analyze_results_session_task \
    import AnalyzeResultsSessionTask
from experimenthost.tasks.completion_service_evaluator_session_task \
    import CompletionServiceEvaluatorSessionTask
from experimenthost.tasks.session_task import SessionTask
from experimenthost.util.candidate_util import CandidateUtil


class ReevaluateCandidateSessionTask(SessionTask):
    """
    SessionTask that performs a re-evaluation of a specific candidate
    given a candidate JSON file specified in the config (for now).

    This task doesn't actually use the Session object that talks
    to the server, but is used within the hierarchy of other SessionTasks
    that do.
    """

    # Tied for Public Enemy #5 for too-many-arguments
    # pylint: disable=too-many-arguments
    def __init__(self, session, master_config, experiment_dir,
                 fitness_objectives, generation, experiment_start_time,
                 experiment_id, completion_service, initial_generation,
                 checkpoint_id=None):
        """
        Constructor.

        :param session: The session with which the task can communicate
                    with the service
        :param master_config: The master config for the task
        :param experiment_dir: The experiment directory for results
        :param fitness_objectives: The FitnessObjectives object
        :param generation: the generation number of the population
        :param experiment_start_time: the experiment start time in seconds
        :param experiment_id: the experiment id
                XXX Can this be derived from experiment_dir?
        :param completion_service: A handle to the CompletionService object
                for performing distributed evaluations.
        :param initial_generation: Flag saying whether or not this is the first
                generation.
        :param checkpoint_id: The checkpoint id (if any) relevant to the task.
        """
        super(ReevaluateCandidateSessionTask, self).__init__(session,
            master_config, experiment_dir, fitness_objectives, checkpoint_id)

        self.generation = generation
        self.experiment_start_time = experiment_start_time
        self.experiment_id = experiment_id
        self.completion_service = completion_service
        self.initial_generation = initial_generation

        self.candidate_util = CandidateUtil(fitness_objectives)


    def run(self):
        """
        Entry point for the session task execution to take over.
        """
        experiment_config = self.master_config.get('experiment_config')

        reevaluate_file = experiment_config.get('reevaluate_chromo')

        assert os.path.exists(reevaluate_file)
        assert experiment_config.get('reevaluate_checkpoint_dir') is None
        assert experiment_config.get('reevaluate_num') < 1000

        candidate_persistence = CandidatePersistence(self.experiment_dir,
                                                 reevaluate_file,
                                                 logger=self.logger)
        orig_candidate = candidate_persistence.restore()

        orig_candidate_id = self.candidate_util.get_candidate_id(orig_candidate)
        print("Re-evaluating chromo %s %s times" % \
              (orig_candidate_id, experiment_config.get('reevaluate_num')))

        copies = []
        counter = 0.001
        # Note: _ is pythonic for unused variable
        for _ in range(experiment_config.get('reevaluate_num')):
            copy = copy.deepcopy(orig_candidate)
            copy['id'] = orig_candidate_id + "." + str(counter)
            copies.append(copy)
            counter += 0.001

        for copy in copies:
            candidate_id = self.candidate_util.get_candidate_id(copy)
            best_candidate_persistence = \
                BestFitnessCandidatePersistence(self.experiment_dir,
                                                candidate_id,
                                                logger=self.logger)
            best_candidate_persistence.persist(copy)

        # XXX There is a mismatch here.
        #     We should not expect ids to always be integers
        use_generation = int(orig_candidate_id)
        self.evaluate_and_analyze_results(copies, use_generation)


    def evaluate_and_analyze_results(self, population, generation_count):

        # Ignore the population_results that is returned.
        # AnalyzeResultsSessionTask works on reading results_dict.json
        # files from the generation directory.
        evaluate_population_task = CompletionServiceEvaluatorSessionTask(
            self.session,
            self.master_config,
            self.experiment_dir,
            self.fitness_objectives,
            generation_count,
            self.experiment_start_time,
            self.experiment_id,
            self.completion_service,
            self.initial_generation,
            population,
            self.checkpoint_id)
        evaluate_population_task.run()

        analyze_results_task = AnalyzeResultsSessionTask(self.session,
                    self.master_config,
                    self.experiment_dir,
                    self.fitness_objectives,
                    self.checkpoint_id)
        analyze_results_task.run()
