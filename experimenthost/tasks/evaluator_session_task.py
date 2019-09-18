
from experimenthost.novelty.novelty_policy_factory \
    import NoveltyPolicyFactory

from experimenthost.regression.fitness_regression_factory \
    import FitnessRegressionFactory

from experimenthost.tasks.session_task import SessionTask

from experimenthost.util.candidate_util import CandidateUtil

from framework.util.experiment_filer import ExperimentFiler


class EvaluatorSessionTask(SessionTask):
    """
    SessionTask that performs an evaluation of a population.

    This task doesn't actually use the Session object that talks
    to the server, but is used within the hierarchy of other SessionTasks
    that do.
    """

    # Public Enemy #1 for too-many-arguments
    # pylint: disable=too-many-arguments
    # Tied for Public Enemy #5 for too-many-instance-attributes
    # pylint: disable=too-many-instance-attributes
    def __init__(self, session, master_config, experiment_dir,
                 fitness_objectives, generation, experiment_start_time,
                 experiment_id, completion_service, initial_generation,
                 population, checkpoint_id=None,
                 novelty_policy=None,
                 server_stats=None):
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
        :param population: The list of candidates to evaluate
        :param checkpoint_id: The checkpoint id (if any) relevant to the task.
        :param novelty_policy: The NoveltyPolicy (if any) relevant to the task
        :param server_stats: Statistics from the ENN Service (if any)
        """
        super(EvaluatorSessionTask, self).__init__(session,
            master_config, experiment_dir, fitness_objectives, checkpoint_id)

        self.generation = generation
        self.experiment_start_time = experiment_start_time
        self.experiment_id = experiment_id
        self.completion_service = completion_service
        self.initial_generation = initial_generation
        self.novelty_policy = novelty_policy
        self.population = population
        self.evaluated_population = None
        self.server_stats = server_stats

        self.candidate_util = CandidateUtil(self.fitness_objectives)
        self.result_update_frequency = 100
        self.timeout_max = 10000000

        # Set up the FitnessRegression policy
        filer = ExperimentFiler(self.experiment_dir)
        regression_archive_file = filer.experiment_file("regression_archive")
        regression_factory = FitnessRegressionFactory()

        experiment_config = self.master_config.get('experiment_config')
        self.fitness_regression = regression_factory.create_fitness_regression(
                                        experiment_config,
                                        self.fitness_objectives,
                                        regression_archive_file)

        # Set up the NoveltyPolicy if none was given
        if self.novelty_policy is None:
            # XXX We use the factory to look at the config, but if
            # we were not sent a novelty_policy, then we should probably
            # just use the NullNoveltyPolicy.  But this is this way to be in
            # the spirit of the original implementation.
            novelty_factory = NoveltyPolicyFactory()
            self.novelty_policy = novelty_factory.create_novelty_policy(
                                            experiment_config,
                                            self.experiment_dir)


    def run(self):
        """
        Entry point for the session task execution to take over.
        """
        self.evaluated_population = self.evaluate(self.population)


    def evaluate(self, population, verbose=False):
        raise NotImplementedError


    def shutdown(self):
        if hasattr(self, 'completion_service') and \
            self.completion_service is not None:

            self.completion_service.shutdown()
