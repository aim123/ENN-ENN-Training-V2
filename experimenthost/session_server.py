#!/usr/bin/env python

from __future__ import print_function

import argparse
import logging
import os
import sys
import time
import traceback

from servicecommon.fitness.fitness_objectives_from_config \
    import FitnessObjectivesFromConfig
from servicecommon.session.enn_population_session_factory \
    import EnnPopulationSessionFactory

from framework.client_script.client import CodeTransfer
from framework.resolver.evaluator_resolver import EvaluatorResolver
from framework.util.experiment_filer import ExperimentFiler
from framework.util.logger import Logger
from framework.util.time_util import get_time


from experimenthost.completion_service.completion_service_shutdown_exception \
    import CompletionServiceShutdownException
from experimenthost.completion_service.completion_service_wrapper \
    import CompletionServiceWrapper

from experimenthost.config.checks import ALL_CHECKS
from experimenthost.config.master_config_reader import MasterConfigReader

from experimenthost.persistence.checkpoint_persistence \
    import CheckpointPersistence
from experimenthost.persistence.master_config_persistence \
    import MasterConfigPersistence
from experimenthost.persistence.experiment_host_error_persistence \
    import ExperimentHostErrorPersistence
from experimenthost.persistence.studio_config_persistence \
    import StudioConfigPersistence

from experimenthost.tasks.analyze_results_session_task \
    import AnalyzeResultsSessionTask
from experimenthost.tasks.get_candidate_session_task \
    import GetCandidateSessionTask
from experimenthost.tasks.list_candidate_ids_session_task \
    import ListCandidateIdsSessionTask
from experimenthost.tasks.local_evaluation_session_task \
    import LocalEvaluationSessionTask
from experimenthost.tasks.reevaluate_best_session_task \
    import ReevaluateBestSessionTask
from experimenthost.tasks.reevaluate_candidate_session_task \
    import ReevaluateCandidateSessionTask
from experimenthost.tasks.run_experiment_session_task \
    import RunExperimentSessionTask

from experimenthost.util.experiment_namer import ExperimentNamer
from experimenthost.util.experiment_name_printer import ExperimentNamePrinter
from experimenthost.util.signal_handler import SignalHandler


class SessionServer():
    """
    Class that sets up and runs the experiment, and orchestrates interaction
    with ENN Session and with workers that perform evaluation

    This class provides the main entry point for running ENN experiments
    and other tasks having to do with experiments.  Its real work is to
    parse command line arguments, read config file(s) and instantiate
    enough classes to initiate the desired SessionTask (see tasks/README.md)
    as divined from the command line arguments.

    This class and the tasks it invokes, provides the glue between the
    evolution code (via Session API) and the distributed evaluation,
    via the completion service.

    If you are looking for the core functionality for evaluation
    and looping through generations, check out the RunExperimentSessionTask
    and the CompletionServiceEvaluatorSessionTask.
    """

    # Public Enemy #1 for too-many-instance-attributes
    # pylint: disable=too-many-instance-attributes
    def __init__(
            self,
            experiment_name=None,
            configpath=None,
            run_experiment=True,
            run_at_construct=True,
            checkpoint_id=None):

        """
        Arguments:

        :param experiment_name: name of the experiment,
               used for checkpointing, naming the results folder,
               and for completion service session name

        :param configpath: path to the config that defines the experiment

        :param run_experiment: whether the experiment is going to be run or
                not when prepare_and_run() method is called
                You may be wondering what is the point of creating the
                SessionServer object otherwise. The answer is it is useful in
                tests to do all the prep work and then run manually a portion
                of the run logic (i.e. run a single generation)

        :param checkpoint_id: id of the checkpoint to restart experiment from

        :return: SessionServer object

        Note that in various call modes (thanks Dan for investigation!)
        run_experiment and run_at_construct are set as follows:
        1. CLI / call to main():
            run_experiment=True
            run_at_construct=False
            (start is delayed because run_with_args method is used instead)

        2. Existing unit tests of session_server
            run_experiment=True
            run_at_construct=True
            (simplest option)

        3. Unit test test_complexity
            run_experiment=False
            run_at_construct=False
            the start is delayed and setup is bypassed because evaluation
            function is overloaded and does not use completion service and
            other prep work

        4. Unit test test_persist_weights
            run_experiment=False
            run_at_construct=True

            The actual run of the experiment is skipped (only prep work done),
            and then experiment is run one epoch at a time.
        """
        # These member variables are set up here for unit tests
        # that want to access the class directly.
        self.experiment_name = experiment_name
        self.config_file = configpath
        self.overlay_config_file = None
        self.run_experiment = run_experiment
        self.checkpoint_id = checkpoint_id
        self.reevaluate_best_candidates = False
        self.analyze_only = False
        self.list_ids = False
        self.get_candidate = None
        self.evaluate_locally = None
        self.completion_service = None

        # Single config dictionary containing key for each of
        # blueprint_config, module_config, builder_config,
        # domain_config and experiment_config.
        # Populated in setup()
        self.master_config = None

        # FitnessObjectives object
        # Populated in setup_from_config()
        self.fitness_objectives = None

        # Handles clean shutdown
        self.signal_handler = None

        # Other member variables initialized later
        self.initial_generation = None
        self.generation = None
        self.checkpoint_persistence = None
        self.last_evaluated_population = None
        self.seen_checkpoint_ids = None
        self.experiment_dir = None
        self.experiment_id = None
        self.experiment_start_time = None
        self.logger = None

        if run_at_construct:
            self.prepare_and_run()


    def prepare_and_run(self):
        self.generation = -1
        self.server_stats = {}

        self.checkpoint_persistence = None
        self.last_evaluated_population = None

        # A list of checkpoint ids that we have come across
        self.seen_checkpoint_ids = []

        self.setup()
        if self.run_experiment:
            self.run()

    def is_needing_completion_service(self):
        need_completion_service = False
        if not self.run_experiment:
            need_completion_service = True
        else:
            need_completion_service = True
            if self.analyze_only \
                or self.list_ids \
                or self.get_candidate is not None \
                or self.evaluate_locally is not None:

                need_completion_service = False
        return need_completion_service


    def setup(self):

        config_reader = MasterConfigReader()
        self.master_config = config_reader.read_master_config(self.config_file,
                                            self.overlay_config_file)

        # Check for any invalid or illegal configuration setting
        for check in ALL_CHECKS:
            check(self)

        self.setup_from_config()
        self.setup_completion_service()


    def setup_from_config(self):
        """
        Sets up experiment directory.
        Assumes config files have been loaded.
        """

        # Name the experiment automatically if necessary
        if self.experiment_name is None:
            namer = ExperimentNamer()
            self.experiment_name = namer.name_this_experiment(self.config_file,
                                                            self.master_config)

        # Spit out the name of the experiment as soon as we know it
        name_printer = ExperimentNamePrinter(self.experiment_name)
        name_printer.print_experiment_name()

        # Register a signal handler
        self.signal_handler = SignalHandler()
        self.signal_handler.append_shutdown_task(name_printer)

        self.experiment_dir = self.determine_experiment_dir(
                                                self.experiment_name)
        self.experiment_id = os.path.basename(self.experiment_dir).replace('_',
                                                                           '-')

        self.checkpoint_persistence = CheckpointPersistence(
                                            folder=self.experiment_dir,
                                            logger=self.logger)

        # Create the FitnessObjectives from the config
        fobj_from_config = FitnessObjectivesFromConfig(self.logger)
        self.fitness_objectives = fobj_from_config.create_fitness_objectives(
                                                self.master_config,
                                                nested_config="blueprint_config")

        # Wait to resolve the Evaluator at least until after we have
        # spit out the experiment name. This allows Keras to print out
        # whatever it will later as not the headline item in the output.
        self.resolve_evaluator()


    def determine_experiment_dir(self, experiment_name):
        """
        Create an experiment directory, if necessary, and also return its name
        """

        experiment_dir = experiment_name

        if self.master_config is not None:
            experiment_config = self.master_config.get('experiment_config')
            basedir = experiment_config.get('experiment_basedir')
            experiment_full_basedir = os.path.abspath(
                                        os.path.expanduser(basedir))
            experiment_dir = os.path.join(experiment_full_basedir,
                                          experiment_dir)
        else:
            experiment_dir = os.path.abspath(os.path.expanduser(experiment_dir))

        if not os.path.exists(experiment_dir):
            os.makedirs(experiment_dir)

        return experiment_dir


    def resolve_evaluator(self):
        """
        Resolve and load code for the evaluator class
        Note we do not actually use the reference here, but it's better
        to find problems before sending things out for distribution.

        :return: An instantiation of the ModelEvaluator class,
                loaded from the various references in the experiment
                and domain config.
        """

        experiment_config = self.master_config.get('experiment_config')
        domain_config = self.master_config.get('domain_config')

        evaluator_resolver = EvaluatorResolver()
        evaluator = evaluator_resolver.resolve(experiment_config.get('domain'),
                class_name=domain_config.get('evaluator_class_name', None),
                evaluator_name=experiment_config.get('network_builder'),
                extra_packages=experiment_config.get('extra_packages'),
                verbose=experiment_config.get('verbose'))

        return evaluator


    def setup_completion_service(self):

        # Setup completion service and other enabled auxiliary structures.
        if not self.is_needing_completion_service():
            return

        # Export configuration to experiment dir
        self.export_master_config()
        studio_config_file = self.export_studio_config()

        experiment_config = self.master_config.get('experiment_config')
        verbose = experiment_config.get('verbose', False)
        code_transfer = CodeTransfer(experiment_config.get('domain'))
        code_transfer.pack_code(self.experiment_dir, verbose=verbose)

        # Be sure the client file exists
        if not os.path.exists(experiment_config.get('client_file')):
            raise ValueError("client_file {0} does not exist".format(
                                    experiment_config.get('client_file')))

        print("experiment id is %s" % self.experiment_id)
        cs_config = experiment_config.get('completion_service')
        self.completion_service = CompletionServiceWrapper(cs_config,
                                                    self.experiment_dir,
                                                    self.experiment_id,
                                                    studio_config_file)
        self.signal_handler.prepend_shutdown_task(self.completion_service)

        self.initial_generation = True
        self.experiment_start_time = time.time()

        archive_file = self.experiment_file("archive")
        domain_config = self.master_config.get('domain_config')
        if domain_config.get('persist_weights', False) and \
            not os.path.exists(archive_file):
            os.mkdir(archive_file)


        # Print configuration for everything
        logger_file = self.experiment_file("experiment_host_log.txt")
        self.logger = Logger(logger_file, "a")
        sys.stdout = self.logger


    def export_master_config(self):
        """
        Export all our configuration information to the experiment dir
        with files that have standardized names.
        These files are not ever read back in again, but are used for debugging
        purposes to see what the SessionServer *thought* the config was.
        """
        master_config_persistence = MasterConfigPersistence(self.experiment_dir,
                                                logger=self.logger)
        master_config_persistence.persist(self.master_config)


    def export_studio_config(self):
        """
        :return: the file reference to the StudioML yaml file to be used.
                    If None, this means to use the StudioML default config
                    policy, which includes looking at ~/.studioml/config.yaml
        """

        # Get the Studio ML config that might have been passed in
        # via the completion service configuration
        experiment_config = self.master_config.get('experiment_config')
        completion_service_config = experiment_config.get('completion_service')
        studio_ml_config = completion_service_config.get('studio_ml_config')

        # See if studio config was put directly in the larger config file
        # as a dictionary.  If this was None or a string file reference,
        # we don't do anything special.
        studio_config_file = studio_ml_config
        if isinstance(studio_ml_config, dict):

            # Write the studio config out to a file, so as to follow existing
            # file-based config ingestion
            persistence = StudioConfigPersistence(self.experiment_dir,
                                    logger=self.logger)
            persistence.persist(studio_ml_config)
            studio_config_file = persistence.get_file_reference()

        return studio_config_file


    def experiment_file(self, filename):
        filer = ExperimentFiler(self.experiment_dir)
        exp_file = filer.experiment_file(filename)
        return exp_file


    def get_latest_checkpoint(self):
        """
        Determine how many generations have been run so far, and how many
        are left.
        """

        checkpoint = -1
        experiment_config = self.master_config.get('experiment_config')
        run_generations = int(experiment_config.get('num_generations'))

        self.seen_checkpoint_ids = self.checkpoint_persistence.restore()
        num_seen = len(self.seen_checkpoint_ids)
        if num_seen == 0:
            return checkpoint, run_generations

        checkpoint = self.seen_checkpoint_ids[num_seen - 1]
        run_generations = run_generations - num_seen

        return checkpoint, run_generations

    # Public Enemy #9 for too-many-branches
    # pylint: disable=too-many-branches
    # Tied for Public Enemy #11 for too-many-statements
    # pylint: disable=too-many-statements
    def run(self):
        try:
            print("")
            print("******************************************************")
            print("*******************STARTING NEW RUN*******************")
            print("******************************************************")
            print(get_time())
            print("ENN SessionServer Class: %s " % self.__class__.__name__)
            print()

            print("experiment base directory is {0}".format(self.experiment_dir))
            print("experiment id is {0}".format(self.experiment_id))

            checkpoint, run_generations = self.get_latest_checkpoint()

            if self.checkpoint_id is None:
                if checkpoint == -1:
                    print("SessionServer no checkpoint found, " +
                          "starting from scratch")
                    self.checkpoint_id = None
                else:
                    self.checkpoint_id = checkpoint
                    print(\
                        "SessionServer resuming from checkpoint at generation ",
                        checkpoint)

            experiment_config = self.master_config.get('experiment_config')

            session_factory = EnnPopulationSessionFactory(self.logger)
            session = session_factory.create_session(
                                            experiment_config.get('enn_service_host'),
                                            experiment_config.get('enn_service_port'),
                                            timeout_in_seconds=None)

            experiment_config = self.master_config.get('experiment_config')

            # Now that we have a session, figure out what we are going to do
            # with this execution of the session server.
            task = None
            if self.analyze_only:
                task = AnalyzeResultsSessionTask(session,
                    self.master_config,
                    self.experiment_dir,
                    self.fitness_objectives,
                    self.checkpoint_id)
            elif self.reevaluate_best_candidates:
                # XXX At some point it would be worthwhile to pass this
                #     reevaluate_chromo in as a command line arg instead of
                #     having to tweak a config.
                if experiment_config.get('reevaluate_chromo') is None:
                    task = ReevaluateBestSessionTask(session,
                        self.master_config,
                        self.experiment_dir,
                        self.fitness_objectives,
                        self.generation,
                        self.experiment_start_time,
                        self.experiment_id,
                        self.completion_service,
                        self.initial_generation,
                        self.checkpoint_id)
                else:
                    task = ReevaluateCandidateSessionTask(session,
                        self.master_config,
                        self.experiment_dir,
                        self.fitness_objectives,
                        self.generation,
                        self.experiment_start_time,
                        self.experiment_id,
                        self.completion_service,
                        self.initial_generation,
                        self.checkpoint_id)
            elif self.list_ids:
                # XXX Eventually the other modes of execution can take this
                #     form to relieve the SessionServer entry point of some
                #     conflation of responsibilities.
                task = ListCandidateIdsSessionTask(session,
                    self.master_config,
                    self.experiment_dir,
                    self.fitness_objectives,
                    self.checkpoint_id)
            elif self.get_candidate is not None:
                task = GetCandidateSessionTask(session,
                    self.master_config,
                    self.experiment_dir,
                    self.fitness_objectives,
                    self.checkpoint_id,
                    self.get_candidate)
            elif self.evaluate_locally is not None:
                task = LocalEvaluationSessionTask(session,
                    self.master_config,
                    self.experiment_dir,
                    self.fitness_objectives,
                    self.checkpoint_id,
                    self.evaluate_locally)
            else:
                task = RunExperimentSessionTask(session,
                    self.master_config,
                    self.experiment_dir,
                    self.fitness_objectives,
                    self.generation,
                    self.experiment_start_time,
                    self.experiment_id,
                    self.completion_service,
                    self.initial_generation,
                    run_generations,
                    self.checkpoint_id)

            if task is not None:
                task.run()

                # Get test results from task, if applicable
                if hasattr(task, 'last_evaluated_population'):
                    self.last_evaluated_population = \
                        task.last_evaluated_population
                if hasattr(task, 'generation'):
                    self.generation = task.generation

        except Exception as exception:
            error = traceback.format_exc()
            timestamp = time.time()

            if not isinstance(exception, CompletionServiceShutdownException):
                # Completion service shutdown is something that normally
                # happens after a ctrl-c.  No need to alert about it.
                error_persistence = ExperimentHostErrorPersistence(
                                                    self.experiment_dir,
                                                    self.generation,
                                                    timestamp,
                                                    logger=self.logger)
                error_persistence.persist(error)
                print(error)

        self.shutdown()


    def shutdown(self):
        self.signal_handler.shutdown()


    def run_with_args(self, argv=None, arg_parser=None):

        self.set_up_logging()

        # Add standard arguments
        class_arg_parser = argparse.ArgumentParser()
        self.add_args(class_arg_parser)

        parents = [class_arg_parser]
        if arg_parser is not None:
            parents.append(arg_parser)

        combined_arg_parser = argparse.ArgumentParser(parents=parents,
                                                     conflict_handler='resolve')

        # Skip the command argument from argv
        args = sys.argv[1:]
        if argv is not None:
            args = argv[1:]

        # Parse the arguments
        # Note: _ is pythonic for unused variable
        namespace, _ = combined_arg_parser.parse_known_args(args)

        self.absorb_args(namespace)
        self.prepare_and_run()

    def add_args(self, class_arg_parser):
        class_arg_parser.add_argument('-n',
            '--experiment_name',
            default=None,
            help="experiment name. Corresponds to a results directory and " +
                 "key for the service.  If none specified, one will be made " +
                 "out of <username>_<domain>_<config>_<datetime>",
            type=str)
        class_arg_parser.add_argument('-c',
            '--config_file',
            default=None,
            help="Path to experiment config file.",
            type=str)
        class_arg_parser.add_argument('-o',
            '--overlay',
            default=None,
            help="Path to overlay config file.",
            type=str)
        class_arg_parser.add_argument('-i',
            '--checkpoint_id',
            default=None,
            help="Checkpoint id from which to resume an experiment",
            type=str)
        class_arg_parser.add_argument('-r',
            '--reevaluate_best_candidates',
            default=False,
            action="store_true",
            help="Reevaluate the best candidate ids from ???")
        class_arg_parser.add_argument('-a',
            '--analyze_only',
            default=False,
            action="store_true",
            help="Analyze results only from ???")
        class_arg_parser.add_argument('-g',
            '--get_candidate',
            default=None,
            help="Gets info about the given candidate id from the given checkpoint")
        class_arg_parser.add_argument('-e',
            '--evaluate_locally',
            default=None,
            help="Locally evaluates the given candidate id from the given checkpoint")
        class_arg_parser.add_argument('-l',
            '--list_ids',
            default=False,
            action="store_true",
            help="List the candidate ids from the given checkpoint")

    def absorb_args(self, namespace):

        self.experiment_name = namespace.experiment_name
        self.config_file = namespace.config_file
        self.overlay_config_file = namespace.overlay
        self.checkpoint_id = namespace.checkpoint_id
        self.reevaluate_best_candidates = namespace.reevaluate_best_candidates
        self.analyze_only = namespace.analyze_only
        self.list_ids = namespace.list_ids
        self.get_candidate = namespace.get_candidate
        self.evaluate_locally = namespace.evaluate_locally

        # We are coming from the command line, so we always want to
        # run something
        self.run_experiment = True

    def set_up_logging(self):
        logging.basicConfig(
            format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
            level=logging.ERROR,
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        self.logger = logging.getLogger('EnnServiceSession')
        self.logger.setLevel(logging.DEBUG)


if __name__ == "__main__":
    # Set up argument parsing
    PARSER = argparse.ArgumentParser(
        "Main entry point to run an ENN Experiment." +
        "usage: python ./session_server.py " +
        "[--config_file=config_file]" +
        "\n" +
        "When capturing output with the linux 'tee' command, consider using " +
        "'tee -i' to allow tee to capture output upon ctrl-c.")

    # When contructing, don't run the experiment just yet.
    # We will do that in run_with_args().
    APP = SessionServer(run_at_construct=False)
    APP.run_with_args(arg_parser=PARSER)
