
from __future__ import print_function

import os
import pprint
import sys
import time

import numpy as np

# Lazily imported in create_progress_bar() below
#from keras.utils import generic_utils

from experimenthost.persistence.evaluation_error_persistence \
    import EvaluationErrorPersistence
from experimenthost.persistence.population_results_persistence \
    import PopulationResultsPersistence
from experimenthost.persistence.results_dict_persistence \
    import ResultsDictPersistence

from experimenthost.tasks.analyze_results_session_task \
    import AnalyzeResultsSessionTask
from experimenthost.tasks.clean_up_archive import CleanUpArchive
from experimenthost.tasks.evaluator_session_task import EvaluatorSessionTask

from experimenthost.vistools.stats_visualizer import StatsVisualizer

from framework.evaluator.data_pathdict import generate_data_pathdict
from framework.persistence.candidate_persistence import CandidatePersistence
from framework.util.time_util import get_time


class CompletionServiceEvaluatorSessionTask(EvaluatorSessionTask):
    """
    SessionTask that performs an evaluation of a population
    by using the Studio ML completion service.

    This task doesn't actually use the Session object that talks
    to the server, but is used within the hierarchy of other SessionTasks
    that do.
    """

    # Public Enemy #1 for too-many-branches
    # pylint: disable=too-many-branches
    # Public Enemy #1 for too-many-locals
    # pylint: disable=too-many-locals
    # Public Enemy #1 for too-many-statements
    # pylint: disable=too-many-statements
    def evaluate(self, population, verbose=False):
        """
        Evaluates the population of the underlying evolutionary algorithm.

        This gets called from our own epoch() method

        For each individual in the population, this method creates a payload
        containing all the information necessary for evaluation, and submits
        it to the completion service.

        It then waits for results from the completion service. Results needed
        for evolution, e.g., fitnesses, are stored in the individuals in the
        population. Other results about the process of the run may be written
        to file or used by auxiliary processes, e.g., regression.

        population -- a list of individuals to be evaluated.
        """
        print("\n%s" % get_time())
        eval_start_time = time.time()

        # Remove persited weights of individuals no longer in the population.
        domain_config = self.master_config.get('domain_config')
        if domain_config.get('persist_weights', False):
            clean_up_archive = CleanUpArchive(self.experiment_dir)
            clean_up_archive.clean_up(population)

        # Some individuals may already have been evaluated,
        # e.g., if we restart a run in the middle of a generation.

        # The ResultsDictPersistence persists/restores a map of
        # candidate id to the worker response dictionary that came back
        # from the worker with evaluation results (if any).
        results_dict_persistence = ResultsDictPersistence(self.experiment_dir,
                                                          self.generation,
                                                          logger=self.logger)
        results_dict = results_dict_persistence.restore()

        #
        # Package and submit each job.
        #
        sys.stdout.flush()
        # sys.stdout.log_enabled = False
        msgs_sent = 0
        num_service_errors = 0
        progbar = self.create_progress_bar(len(population))

        experiment_config = self.master_config.get('experiment_config')
        domain_config = self.master_config.get('domain_config')

        # Submit only the necessary tasks to the completion service
        for candidate in population:

            candidate_id = self.candidate_util.get_candidate_id(candidate)
            candidate_fitness = self.candidate_util.get_candidate_fitness(
                                    candidate)
            if candidate_id in results_dict:
                msgs_sent += 1
                progbar.add(1)
                continue
            elif not experiment_config.get('evaluate_all') and \
                    candidate_fitness is not None and \
                    not self.initial_generation:
                continue

            # Check for candidate-specific errors from the service
            interpretation = candidate.get('interpretation', None)
            if interpretation is None:
                print("Not queueing {0}. No interpretation in candidate.".\
                      format(candidate_id))

                identity = candidate.get('identity', None)
                if identity is None:
                    # Allow for old-style candidates, too
                    identity = candidate
                error = identity.get('error', None)
                num_service_errors = num_service_errors + 1

                if error is not None:
                    print("Error info:")
                    for key, value in list(error.items()):
                        print("{0}: {1}".format(key, value))
                continue


            # Package files required for evaluation into file_dict.
            # Note: Studio might end up transforming some of the references
            #       to these files in remote evaluation.
            file_dict = {}
            file_dict['lib'] = os.path.join(self.experiment_dir,
                                           "worker_code_snapshot.tar.gz")
            if domain_config.get('send_data', True):
                data_pathdict = generate_data_pathdict(domain_config,
                                                       convert_urls=False)
                file_dict.update(data_pathdict)

            # Package the core payload into the payload dictionary:
            # worker_request_dict.
            time_now = time.time()
            worker_request_dict = {
                'config': {
                    'domain': experiment_config.get('domain'),
                    'domain_config': domain_config,
                    'evaluator': experiment_config.get('network_builder'),
                    'extra_packages': experiment_config.get('extra_packages'),
                    'resources_needed': experiment_config.get('cs_resources_needed')
                },
                'id': candidate_id,
                'identity': {
                    'experiment_id': self.experiment_id,
                    'experiment_timestamp': self.experiment_start_time,
                    'generation': self.generation,
                    'generation_timestamp': eval_start_time,
                    'submit_timestamp': time_now
                },
                'interpretation': interpretation
            }

            # Finally, submit the task.
            self.completion_service.submit_task_with_files(
                self.experiment_id,
                experiment_config.get('client_file'),
                worker_request_dict,
                file_dict,
                job_id=candidate_id
            )
            msgs_sent += 1
            progbar.add(1)

            # End of sending tasks to completion service

        sys.stdout.log_enabled = True

        # Set up some lists for per-candidate stats
        stats = {
            'fitnesses': [],
            'elapsed_times': [],
            'num_epochs_trained': [],
            'queue_wait_times': []
        }

        msgs_received = 0

        # Set up some dictionaries:
        # 1. candidate_dict is id -> unevaluated candidate from candidate
        #    received
        # 2. evaluated_candidate_dict is id -> evaluated candidate handed
        #    back to service
        candidate_dict = {}
        evaluated_candidate_dict = {}
        for candidate in population:
            id_key = self.candidate_util.get_candidate_id(candidate)

            candidate_dict[id_key] = candidate

            if id_key in results_dict:
                # We already have results for this candidate.
                # This might be able to happen when a local checkpoint is
                # re-read.
                # Add the fitness as already computed to our records.
                worker_response_dict = results_dict[id_key]
                evaluated_candidate = self._create_evaluated_candidate_dict(
                    worker_response_dict, candidate, stats)
                evaluated_candidate_dict[id_key] = evaluated_candidate
                msgs_received += 1

        # We normally expect that *all* candidates from the service are
        # constructed with no errors, but problems internal to the service can
        # result in small temporary blips of bad populations.  While we work
        # to minimize these blips, that doesn't mean they don't happen.
        #
        # The good_service_population_ratio below is a number implying that we
        # need X percentage of the population coming back from the service to
        # have no errors in their construction in order to consider the
        # population robust for its generation to be evaluated.
        good_service_population_ratio = 0.8
        bad_service_population_ratio = (1.0 - good_service_population_ratio)
        pop_size = len(population)
        if num_service_errors >= pop_size * bad_service_population_ratio:
            raise ValueError("Most of the population had service errors.")

        # If no messages are sent, then we move to next generation
        # We still visualize the best from the population even if no results
        # are sent back
        if msgs_sent == 0:
            print("warning: no messages sent for gen %s" % self.generation)

            # Translate the existing population's fitness results dictionary
            # into a list of only evaluated_population.
            evaluated_population = list(evaluated_candidate_dict.values())
            self.maybe_visualize_best(evaluated_population)
            return evaluated_population

        print("sent {0} messages for gen {1} at {2}".format(
                    msgs_sent, self.generation, get_time()))

        # Prepare to collect results.
        sys.stdout.flush()
        sys.stdout.log_enabled = False
        last_modified_time = 0
        bad_msgs_received = 0
        good_msgs_received = 0
        progbar = self.create_progress_bar(msgs_sent)

        #
        # Collect results until all are received.
        #

        # Wait for results of any candidate's evaluation to come back from
        # completion service.
        while msgs_received < msgs_sent:
            start_time = time.time()
            timeout_seconds = self.choose_timeout(msgs_received, msgs_sent)

            # Grab results in the form of results dictionary:
            # worker_response_dict.
            worker_response_dict = \
                self.completion_service.get_results_with_timeout(
                        timeout_seconds)

            # Break if timeout reached.
            elapsed_time = time.time() - start_time
            if elapsed_time >= timeout_seconds:
                # assert worker_response_dict is None
                sys.stdout.log_enabled = True
                print("Timeout occurred, moving on to next generation!")
                print("Elapsed Time: %d, Timeout Seconds: %d" % \
                      (int(elapsed_time), int(timeout_seconds)))
                print("Timeout settings: %s" \
                        % str(experiment_config.get('timeout_settings')))
                break

            # Perform some checks on the received results dictionary and do some
            # logging and/or write results to file accordingly.
            sys.stdout.log_enabled = True
            if not isinstance(worker_response_dict, dict):
                if worker_response_dict is not None:
                    print("\nReturned results not in correct format, has type %s" % \
                        type(worker_response_dict))
                    pprint.pprint(worker_response_dict)
                    count_msg = False
                    write_to_file = True
                    skip_to_next = True
                else:
                    print("\nNone returned, maybe darkcycle error?")
                    print("Elapsed seconds: %s Timeout seconds: %s" % \
                          (elapsed_time, timeout_seconds))
                    count_msg = False
                    write_to_file = False
                    skip_to_next = True
            elif worker_response_dict.get('identity', {}).\
                    get('generation_timestamp', None) != eval_start_time:
                print("\nReturned results generation timestamp does not match (ignored)!")

                identity = worker_response_dict.get('identity', {})
                if identity.get('generation', None) != self.generation:
                    print("Returned results from different generation (ignored)!")
                pprint.pprint(worker_response_dict)
                count_msg = False
                write_to_file = False
                skip_to_next = True
            elif worker_response_dict.get('metrics', {}).\
                    get('execution', {}).\
                    get('eval_error', None) is not None:
                print("\nReturn results has chromosome evaluation error!")
                pprint.pprint(worker_response_dict)
                count_msg = False
                write_to_file = True
                skip_to_next = True
            else:  # No errors or issues with returned results
                count_msg = True
                write_to_file = False
                skip_to_next = False
            sys.stdout.log_enabled = False

            if count_msg:
                good_msgs_received += 1
            else:
                bad_msgs_received += 1
            msgs_received += 1
            progbar.add(1)

            candidate_id = worker_response_dict.get('id', None) \
                    if isinstance(worker_response_dict, dict) else None

            if write_to_file:
                timestamp = time.time()
                error_persistence = EvaluationErrorPersistence(self.experiment_dir,
                                            self.generation, candidate_id,
                                            timestamp,
                                            logger=self.logger)
                error_persistence.persist(worker_response_dict)

            if skip_to_next:
                continue

            #
            # Get the fitness and other information from the results dictionary.
            #
            metrics = worker_response_dict.get('metrics', {}) \
                    if isinstance(worker_response_dict, dict) else None
            execution = metrics.get('execution', {})
            execution['return_timestamp'] = time.time()

            id_key = str(candidate_id)
            candidate = candidate_dict[id_key]

            evaluated_candidate = self._create_evaluated_candidate_dict(
                worker_response_dict, candidate, stats)
            evaluated_candidate_dict[id_key] = evaluated_candidate

            # If using regression, extract features for regression.
            self.fitness_regression.add_sample(id_key, metrics)

            # Write candidate to generation directory
            candidate_persistence = CandidatePersistence(
                                            self.experiment_dir,
                                            candidate_id=id_key,
                                            generation=self.generation,
                                            logger=self.logger)
            candidate_persistence.persist(evaluated_candidate)

            # Substitute out the file reference
            # XXX Why do we do this?
            candidate_filename = candidate_persistence.get_file_reference()
            worker_response_dict['interpretation'] = candidate_filename

            # Checkpoint the results for this generation.
            results_dict[id_key] = worker_response_dict
            if time.time() - last_modified_time > self.result_update_frequency or \
                    msgs_received == msgs_sent:
                results_dict_persistence = ResultsDictPersistence(
                                                    self.experiment_dir,
                                                    self.generation,
                                                    logger=self.logger)
                results_dict_persistence.persist(results_dict)
                last_modified_time = time.time()

        # Translate the existing population's fitness results dictionary
        # into a list of only evaluated_population.
        evaluated_population = list(evaluated_candidate_dict.values())

        sys.stdout.log_enabled = True

        #
        # Now that all results have been received, update auxiliary
        # systems that are being used: novelty and/or regression.
        #
        self.novelty_policy.update(evaluated_candidate_dict)
        self.fitness_regression.update(results_dict, evaluated_candidate_dict)

        #
        # Wrap up generation. Log and save interesting information about
        # the generation.
        #
        if min(len(stats.get('elapsed_times')),
               len(stats.get('fitnesses')),
               len(stats.get('num_epochs_trained'))) > 0:

            print("elapsed time/fitness/epochs trained:")
            elapsed_times = stats.get('elapsed_times')
            fitnesses = stats.get('fitnesses')
            num_epochs_trained = stats.get('num_epochs_trained')
            queue_wait_times = stats.get('queue_wait_times')

            combined = list(zip(elapsed_times, fitnesses, num_epochs_trained))
            print(sorted(combined, key=lambda x: x[0]))

            # Numpy sometimes returns types that are not JSON serializable.
            # Be sure we get serializable values back
            meant = float(np.mean(elapsed_times))
            stdt = float(np.std(elapsed_times))
            medt = float(np.median(elapsed_times))
            maxt = int(np.max(elapsed_times))
            totalt = float(time.time() - eval_start_time)
            trpt = float(meant / totalt)
            meanq = float(np.mean(queue_wait_times))
            maxq = int(np.max(queue_wait_times))
            stdq = float(np.std(queue_wait_times))
            succ_rate = float(good_msgs_received) / msgs_sent

            print("mean individual evaluation time:", round(meant, 3))
            print("median individual evaluation time:", round(medt, 3))
            print("std of individual evaluation time:", round(stdt, 3))
            print("mean throughput:", round(trpt, 3))
            print("Population evaluation time: ", round(totalt, 3))
            if experiment_config.get('visualize'):
                stats_visualizer = StatsVisualizer(self.experiment_dir)
                stats_visualizer.record_and_visualize_stats(
                    (meant, stdt, medt, trpt, totalt, maxt, meanq, maxq, stdq, \
                    succ_rate), self.server_stats)
        elif msgs_received == 0:
            print("warning: no valid messages returned for gen %s" \
                  % (self.generation))
            print("%d invalid messages were returned" % bad_msgs_received)
            if experiment_config.get('no_results_quit'):
                self.shutdown()
                sys.exit()

        self.maybe_visualize_best(evaluated_population)

        evaluated_population_persistence = PopulationResultsPersistence(
                                                    self.experiment_dir,
                                                    self.generation,
                                                    logger=self.logger)
        evaluated_population_persistence.persist(evaluated_population)

        # Now that the population has been evaluated, and all the information
        # about this generation has been recorded,
        # we can let evolution continue.
        return evaluated_population


    def _create_evaluated_candidate_dict(self, worker_response_dict,
                                         candidate, stats):
        """
        Helper function to package up fitness results for a particular
        candidate and add to running stats.
        """

        # Create the new Evaluated Candidate from the original which
        # came down from the service.  Everything is the same except for
        # the metrics, which comes directly from the domain.

        candidate_id = worker_response_dict.get('id', None)
        metrics = worker_response_dict.get('metrics', None)

        evaluated_candidate = {
            'id': candidate_id,
            'identity': candidate.get('identity', None),
            'interpretation': candidate.get('interpretation', None),
            'metrics': metrics
        }

        if metrics is None:
            metrics = {}

        #
        # Add to the stats
        #

        total_num_epochs_trained = metrics.get('total_num_epochs_trained', 0)
        if total_num_epochs_trained is None:
            total_num_epochs_trained = 0

        fitness = self.candidate_util.get_candidate_fitness(evaluated_candidate)
        if fitness is None:
            fitness = 0.0

        # If using a multi-objective approach, determine the second objective.
        # XXX alt_objective doesn't get used
        alt_objective = self.candidate_util.get_candidate_fitness(
                                            evaluated_candidate,
                                            fitness_objective_index=1)

        experiment_config = self.master_config.get('experiment_config')
        if experiment_config.get('age_layering'):
            alt_objective = total_num_epochs_trained

        # XXX This is dubious that alt_obj is being supplanted
        alt_objective = self.novelty_policy.compute_novelty(metrics,
                                                            alt_objective)

        # Add to collected information about this generation.
        stats.get('fitnesses').append(fitness)

        execution = metrics.get('execution', {})

        client_elapsed_time = int(execution.get('client_elapsed_time', 0))
        stats.get('elapsed_times').append(client_elapsed_time)

        queue_wait_time = int(execution.get('queue_wait_time', 0))
        stats.get('queue_wait_times').append(queue_wait_time)

        stats.get('num_epochs_trained').append(total_num_epochs_trained)

        return evaluated_candidate


    def maybe_visualize_best(self, evaluated_population):

        experiment_config = self.master_config.get('experiment_config')
        if experiment_config.get('visualize') and \
            evaluated_population is not None and \
            len(evaluated_population) > 0:

            best_candidate = None

            # Ranking comparators put more fit candidates closer to the
            # start of the list when sorting.  That is, better values
            # are *lower*.  The FitnessObjectives object takes care of
            # the sense of the comparator for maximization/minimization
            # from the config when it is set up.
            comparator = self.fitness_objectives.get_ranking_comparator(0)
            for candidate in evaluated_population:
                if best_candidate is None:
                    best_candidate = candidate
                else:
                    candidate_fitness = \
                        self.candidate_util.get_candidate_fitness(candidate)
                    best_fitness = \
                        self.candidate_util.get_candidate_fitness(
                                                            best_candidate)
                    comparison = comparator.compare(candidate_fitness,
                                                    best_fitness)
                    # For ranking comparators, better values are smaller.
                    if comparison < 0:
                        best_candidate = candidate

            analyze_results_task = AnalyzeResultsSessionTask(self.session,
                    self.master_config,
                    self.experiment_dir,
                    self.fitness_objectives,
                    self.checkpoint_id)
            analyze_results_task.draw_best_candidate_results(best_candidate,
                                                 generation=self.generation)

        sys.stdout.flush()


    def choose_timeout(self, msgs_received, msgs_sent):
        """
        Compute the number of seconds until timeout.
        """

        fraction_completed = float(msgs_received) / float(msgs_sent)

        experiment_config = self.master_config.get('experiment_config')
        timeout_settings = experiment_config.get('timeout_settings')

        sorted_timeouts = sorted(timeout_settings, reverse=True,
                                 key=lambda x: x[0])
        for fraction, timeout_value in sorted_timeouts:

            if fraction_completed >= fraction:
                return timeout_value

        return self.timeout_max


    def create_progress_bar(self, num_iterations, interval=0.0):

        # Lazy import of Keras to prevent output so early
        from keras.utils import generic_utils
        progbar = generic_utils.Progbar(num_iterations, interval=interval)
        return progbar
