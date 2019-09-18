
from __future__ import print_function

from future.builtins import range

from experimenthost.novelty.novelty_policy_factory \
    import NoveltyPolicyFactory

from experimenthost.persistence.checkpoint_persistence \
    import CheckpointPersistence
from experimenthost.persistence.softorder_persistor import SoftOrderPersistor

from experimenthost.tasks.completion_service_evaluator_session_task \
    import CompletionServiceEvaluatorSessionTask
from experimenthost.tasks.random_fitness_mock_evaluator_session_task \
    import RandomFitnessMockEvaluatorSessionTask
from experimenthost.tasks.population_response_util import PopulationResponseUtil
from experimenthost.tasks.session_task import SessionTask


class RunExperimentSessionTask(SessionTask):
    """
    SessionTask that performs the running of multiple generations
    for an experiment.
    """

    # Tied for Public Enemy #3 for too-many-arguments
    # pylint: disable=too-many-arguments
    # Tied for Public Enemy #5 for too-many-instance-attributes
    # pylint: disable=too-many-instance-attributes
    def __init__(self, session, master_config, experiment_dir,
                 fitness_objectives, generation, experiment_start_time,
                 experiment_id, completion_service, initial_generation,
                 run_generations,
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
        :param run_generations: Number of generations to run for
        :param checkpoint_id: The checkpoint id (if any) relevant to the task.
        """
        super(RunExperimentSessionTask, self).__init__(session,
            master_config, experiment_dir, fitness_objectives, checkpoint_id)

        self.generation = generation
        self.experiment_start_time = experiment_start_time
        self.experiment_id = experiment_id
        self.completion_service = completion_service
        self.initial_generation = initial_generation
        self.run_generations = run_generations

        self.population_response_util = PopulationResponseUtil()

        # These are fields to be populated by unpack_response()
        experiment_config = self.master_config.get('experiment_config')
        self.persistor = SoftOrderPersistor(self.experiment_dir,
                                            self.fitness_objectives,
                                            draw=experiment_config.get('visualize'),
                                            logger=self.logger)
        self.server_stats = {}

        self.checkpoint_persistence = CheckpointPersistence(
                                            folder=self.experiment_dir,
                                            logger=self.logger)
        self.seen_checkpoint_ids = self.checkpoint_persistence.restore()

        # This is used to pass test results back to the tests themselves
        self.last_evaluated_population = None

        novelty_factory = NoveltyPolicyFactory()
        self.novelty_policy = novelty_factory.create_novelty_policy(experiment_config,
                                                                    self.experiment_dir)


    def run(self):
        """
        Entry point for the session task execution to take over.
        """
        self.epoch(self.run_generations)


    def epoch(self, run_generations):
        """
        Runs evolution for n epochs
        """

        population_response = {}
        if self.checkpoint_id is not None:
            population_response = self.session.get_population(
                                        self.experiment_dir,
                                        self.checkpoint_id)
            population_response = self.process_one_epoch(population_response)

        # Only send what the service needs
        service_config = {
            'blueprint_config': self.master_config.get('blueprint_config'),
            'builder_config': self.master_config.get('builder_config'),
            'module_config': self.master_config.get('module_config'),
        }

        # Note: _ is pythonic for unused variable
        for _ in range(run_generations):

            population_results = population_response.get("population", [])
            if not any(population_results):
                population_response = None

            population_response = self.session.next_population(
                                                self.experiment_dir,
                                                service_config,
                                                population_response)

            population_response = self.process_one_epoch(population_response)

        print("all done, goodbye")


    def process_one_epoch(self, population_response):

        # Disassemble the returned population response
        population = self.population_response_util.unpack_response(
                                    population_response, self)

        # Only persist the seen checkpoint ids when we are training,
        # not during reevaluate_best()
        self.checkpoint_persistence.persist(self.seen_checkpoint_ids)

        if self.checkpoint_id is None:
            print("Processing initial generation")
        else:
            print("Processing generation at checkpoint %s" % self.checkpoint_id)

        # This did have a try around it, but looked only for arg count
        # Args have changed, so unclear if this was significant
        population = self.evaluate(population)

        self.persistor.persist(population, self.generation)

        # Keep state for ourselves as to the results from the last population
        # we worked on.  While the Service does keep state as to the population,
        # what it tells us is the *next* batch to work on, not what we already
        # did.  We have that information, so we use it for testing fitness comes
        # in a certain range (see session_server_test.py).
        self.last_evaluated_population = population

        # Incrementing the generation is a bit touchy in terms of getting
        # the locally produced files in sync with the checkpoint number on
        # the other side of the service.
        self.generation += 1

        # Prepare the population response for next generation
        population_response = self.population_response_util.pack_response(
                                            population, self)

        return population_response


    def evaluate(self, population, verbose=False):

        experiment_config = self.master_config.get('experiment_config')
        experiment_host_evaluator = experiment_config.get(
                                                'experiment_host_evaluator',
                                                None)

        if experiment_host_evaluator == "RandomFitnessMock":
            evaluator_session_task = RandomFitnessMockEvaluatorSessionTask(
                self.session,
                self.master_config,
                self.experiment_dir,
                self.fitness_objectives,
                self.generation,
                self.experiment_start_time,
                self.experiment_id,
                self.completion_service,
                self.initial_generation,
                population,
                self.checkpoint_id,
                self.novelty_policy)
        else:
            # Default is to use the CompletionService
            evaluator_session_task = CompletionServiceEvaluatorSessionTask(
                self.session,
                self.master_config,
                self.experiment_dir,
                self.fitness_objectives,
                self.generation,
                self.experiment_start_time,
                self.experiment_id,
                self.completion_service,
                self.initial_generation,
                population,
                self.checkpoint_id,
                self.novelty_policy,
                server_stats=self.server_stats)

        population = evaluator_session_task.evaluate(population, verbose)
        self.initial_generation = False
        return population
