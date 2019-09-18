
from __future__ import print_function

import os
import numpy as np

from experimenthost.persistence.best_fitness_candidate_persistence \
    import BestFitnessCandidatePersistence
from experimenthost.persistence.checkpoint_persistence import CheckpointPersistence
from experimenthost.persistence.results_dict_persistence \
    import ResultsDictPersistence
from experimenthost.persistence.softorder_persistor import SoftOrderPersistor

from experimenthost.tasks.reevaluate_candidate_session_task \
    import ReevaluateCandidateSessionTask
from experimenthost.tasks.population_response_util import PopulationResponseUtil
from experimenthost.tasks.session_task import SessionTask

from experimenthost.util.candidate_util import CandidateUtil


class ReevaluateBestSessionTask(SessionTask):
    """
    SessionTask that performs a re-evaluation of the best candidates
    from each generation.
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
        super(ReevaluateBestSessionTask, self).__init__(session,
            master_config, experiment_dir, fitness_objectives, checkpoint_id)

        self.generation = generation
        self.experiment_start_time = experiment_start_time
        self.experiment_id = experiment_id
        self.completion_service = completion_service
        self.initial_generation = initial_generation

        self.candidate_util = CandidateUtil(fitness_objectives)
        self.population_response_util = PopulationResponseUtil()

        # These are fields to be populated by unpack_response()
        experiment_config = self.master_config.get('experiment_config')
        self.persistor = SoftOrderPersistor(self.experiment_dir,
                                            self.fitness_objectives,
                                            draw=experiment_config.get('visualize'),
                                            logger=self.logger)
        self.server_stats = {}
        self.seen_checkpoint_ids = []


    def run(self):
        """
        Entry point for the session task execution to take over.
        """

        experiment_config = self.master_config.get('experiment_config')
        assert os.path.exists(experiment_config.get('reevaluate_checkpoint_dir'))
        print("Re-evaluating top %s chromosomes found from experiment %s" % \
              (experiment_config.get('reevaluate_num'),
               experiment_config.get('reevaluate_checkpoint_dir')))

        candidate_fit_dict = {}

        # Read in the contents of the checkpoint_ids.txt file which contains
        # all references to any checkpoint training has seen.
        # By convention reevalute_checkpoint_dir is where this file is coming
        # from, and self.checkpoint_dir is where new results are being
        # written to.
        restoring_checkpoint_persistence = CheckpointPersistence(
                    folder=experiment_config.get('reevaluate_checkpoint_dir'),
                    logger=self.logger)
        self.seen_checkpoint_ids = restoring_checkpoint_persistence.restore()

        for checkpoint_id in self.seen_checkpoint_ids:

            print("Analyzing chromos in %s" % checkpoint_id)

            population_response = self.session.get_population(
                experiment_config.get('reevaluate_checkpoint_dir'),
                checkpoint_id)
            pop = self.population_response_util.unpack_response(
                    population_response, self)

            for candidate in pop:
                id_key = self.candidate_util.get_candidate_id(candidate)

                # Get the persisted Worker Results dictionaries
                results_dict_persistence = ResultsDictPersistence(
                    experiment_config.get('reevaluate_checkpoint_dir'),
                    self.generation,
                    logger=self.logger)
                results_dict = results_dict_persistence.restore()

                candidate_fitness = None
                if any(results_dict):
                    if id_key in results_dict:
                        candidate_results_dict = results_dict[id_key]
                        # This is not quite a candidate, but the get-mechanism
                        # should be the same
                        candidate_fitness = \
                            self.candidate_util.get_candidate_fitness(
                                candidate_results_dict)
                if candidate_fitness is None:
                    candidate_fitness = 0.0

                if id_key not in candidate_fit_dict:
                    candidate_fit_dict[id_key] = {'candidate': candidate,
                                                  'fit': [candidate_fitness]}
                else:
                    candidate_fit_dict[id_key]['candidate'] = candidate
                    candidate_fit_dict[id_key]['fit'].append(candidate_fitness)

        avg = [(x['candidate'], np.mean(x['fit'])) \
                for x in list(candidate_fit_dict.values())]
        best = sorted(avg, key=lambda x: x[1],
                      reverse=True)[:experiment_config.get('reevaluate_num')]
        best_candidates = [x[0] for x in best]
        best_candidate_ids = [self.candidate_util.get_candidate_id(x[0]) \
                                for x in best]
        best_fit = [round(x[1], 4) for x in best]

        if len(best_candidates) == 0:
            print("No chromos found, doing nothing")
            return

        for candidate in best_candidates:
            candidate_id = self.candidate_util.get_candidate_id(candidate)
            best_candidate_persistence = BestFitnessCandidatePersistence(
                                                self.experiment_dir,
                                                candidate_id,
                                                logger=self.logger)
            best_candidate_persistence.persist(candidate)

        print("Best chromos:")
        print(list(zip(best_candidate_ids, best_fit)))
        print("Best chromo stats:")
        print("Min: %s Mean: %s Max: %s Std: %s" % \
              (round(np.min(best_fit), 4), round(np.mean(best_fit), 4),
               round(np.max(best_fit), 4), round(np.std(best_fit), 4)))

        # We use generation + 1 for reporting here because we are really
        # composing a population of the best candidates across many
        # different previous generations, and as such doesn't really
        # correspond to any generation number of the past.
        reevaluate_candidate_task = ReevaluateCandidateSessionTask(\
            self.session,
            self.master_config,
            self.experiment_dir,
            self.fitness_objectives,
            self.generation,
            self.experiment_start_time,
            self.experiment_id,
            self.completion_service,
            self.initial_generation,
            self.checkpoint_id)
        reevaluate_candidate_task.evaluate_and_analyze_results(best_candidates,
                                                            self.generation + 1)
