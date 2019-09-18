
import os

from experimenthost.util.bucket_name_filter import BucketNameFilter
from experimenthost.util.queue_name_filter import QueueNameFilter

from framework.domain.config import Config


class ExperimentConfig():
    """
    Class for highest level configs for running an ENN experiment.

    This config includes parameters that are applied at the experiment level,
    and points to more specific configs, e.g., for the evolution algorithm,
    domain, and network builder.
    """

    def build_config(self, dict_reference):
        """
        Called by the Session Server to build up the Experiment Host
        configuration.
        :param dict_reference: A reference to a dictionary to use as a basis.
                This can either be a filename reference or a dictionary itself.
        :return: an "outer" config dictionary that has a single
                "experiment_config" key filled in with what was read from the
                dict_reference, with any missing defaults filled in.
        """

        # First read the config as-is
        config_loader = Config()
        loaded_config = config_loader.read_config(dict_reference,
                                                  default_config=None)

        verbose = loaded_config.get('verbose', False)

        # See if this is an all-in-one config so we know which
        # dictionary to apply the defaults to.
        outside_config = {}
        experiment_config = loaded_config.get('experiment_config', None)
        if experiment_config is not None:
            outside_config = loaded_config
        else:
            outside_config['experiment_config'] = loaded_config

        # Get our defaults and apply them to the right dict
        default_config = self._get_default_config()
        outside_config['experiment_config'] = config_loader.update(
                                            default_config,
                                            outside_config['experiment_config'],
                                            verbose=verbose)

        # Migrate some keys to new sub-sections for compatibility
        experiment_config = outside_config['experiment_config']

        # Completion Service subsection compatibility
        experiment_config = config_loader.key_prefix_to_subsection(
                                                    experiment_config,
                                                    'cs_',
                                                    'completion_service')

        experiment_config = self.filter_bucket_names(experiment_config)
        experiment_config = self.filter_queue_names(experiment_config)

        outside_config['experiment_config'] = experiment_config

        return outside_config


    def _get_default_config(self):
        """
        :return: The default experiment_config.
        """

        # Name/path of script to run on worker machines
        # This goes in the config as a default
        client_file = os.path.join(
                os.path.abspath(os.path.dirname(__file__)),
                "../../framework/client_script/client.py"
            )

        default_config = {

            # Whether to enable "age layering" or not
            'age_layering': False,

            # Novelty search params
            # Whether to enable novelty search or not
            'novelty_search': False,
            # Maximum archive size
            'novelty_max_size': 1000,
            # Number of nearest neighbors to use for computing novelty
            'novelty_k': 7,
            # Probability of adding an individual to archive
            'novelty_p': 0.15,

            # On-line regression
            # Whether to use online regression system.
            'online_regression': False,
            # You need at least <threshold> samples before you fit the
            # regression model.
            'online_regression_sample_threshold': 20,
            # You want to predict the accuracy at <target> epochs.
            'online_regression_max_target_epoch': 50,
            # The metric you want to project
            # Options for target_metric:
            # "Bleu_*" (where * is 1, 2, 3, or 4), "ROUGE_L", "METEOR", "CIDEr",
            # "Loss"
            'online_regression_target_metric': 'fitness',

            # Base directory for experiment files
            'experiment_basedir': "../results/",

            # Specifies which class represents the 'domain'.
            # A 'domain' describes the different files and dimensions to use to
            # train a model. This property is used to dynamically import the
            # classes for ModelEvaluators and DomainrConfig.
            'domain': "omniglot",
            # Dictionary specification or configuration file name for
            # problem domain, None for default
            'domain_config': None,
            # Legacy Configuration file for problem domain, none for default
            'domain_config_in': None,
            # Domains can set this in the experiment config to get out from
            # under legacy file/class naming conventions. This should be
            # a string containing a class name which can be converted to
            # snake-case to figure out the filename.
            'domain_config_class_name': None,

            # Legacy configuration file(s) for evolution,
            # None for default
            'evolution_algorithm_config_in': None,
            # Dictionary specification or configuration file name for
            # service blueprint (top-level of coevolution). None for default
            'blueprint_config': None,
            # Dictionary specification or configuration file name for
            # service module (low-level of coevolution). None for default
            'module_config': None,

            # Legacy: partial class name for evaluators
            # Use 'evaluator_class_name' on DomainConfig instead.
            'network_builder': None,
            # Dictionary specification or configuration file name for
            # service builder. None for default
            'builder_config': None,
            # Legacy Configuration file for evolution,
            # empty string for default
            'network_builder_config_in': None,

            # Name/path of script to run on worker machines
            'client_file': client_file,

            # This is the name of the method to use in visualization.py to
            # generate a .png image of the model generated from a chromosome:
            # * 'DefaultKerasNetworkVisualizer' uses Keras default model
            #       representation using keras.plot_model
            # * 'SeeNN' gives a nice colorful full-connectivity view of a
            #       candidate's model. Impressive for its complexity.
            # * 'SeeNNBlueprint' gives a high-level module connectivity view
            #       of a candidate's model. Good for seeing highest-level
            #       blueprint evolution of module connectivity.
            # * 'SeeNNModuleCatalog' gives a mid-level layer connectivity list
            #       for each module in a candidate's model. Good for seeing
            #       the repeating layer elements in a candidate's network.
            'network_visualization': [
                        "SeeNN",
                        "SeeNNBlueprint",
                        "SeeNNModuleCatalog"
                    ],
            # Whether to draw stats and best networks or not
            'visualize': True,

            # Number of generations to run evolution
            'num_generations': 60,
            # Whether to evaluate all chromosomes at every generation
            'evaluate_all': True,

            # Settings for timing out, is a list of percentage complete and wait
            # time. For instance, if we are 50% done evaluating the genes in the
            #  population, we wait for 14,400 seconds. If we are 90% done,
            # we wait for 3,000 seconds only.
            'timeout_settings': [(0.5, 14400), (0.9, 3000), (0.95, 2000),
                                 (0.97, 1000)],
            # If we receive no valid results back from completion service,
            # we kill the experiment.
            'no_results_quit': True,

            # Number of best to re-evaluate
            'reevaluate_num': 50,
            # Experiment directory from which to get evolution checkpoints
            'reevaluate_checkpoint_dir': None,
            # Whether to reevaluate a particular chromo, or just the top best
            # chromo
            'reevaluate_chromo': None,

            # These are the configuration for completion service
            'completion_service': {

                # Should we clean up the queue and other studio files?
                # False (default) is good for regular data science
                # True is good for repetitive testing.
                'cleanup': False,

                # Default timeout for workers
                'timeout': 1000,

                # Whether to run locally or not
                'local': True,

                # Type of worker to use, choose from 'gcloud', 'gcspot',
                # 'ec2', 'ec2spot' or None for local usage
                'cloud': None,

                # Number of workers to spin-up
                'num_workers': 1,

                # Bid for workers on AWS
                'bid': "30%",

                # Keypair to SSH into workers on AWS
                'ssh_keypair': "enn-us-east",
                # 'ssh_keypair': None,

                # Studioml yaml config file to use.
                # There are 3 types of settings:
                #   1. None, for default behavior of studio getting its config
                #      from ~/.studioml/config.yaml
                #   2. A string file name.  Studio will use this file instead
                #       of the default (like 1 above)
                #   3. a fully specified dictionary structure comprising the
                #      entire studio ML config to use.
                #
                #      Note that different ENN config file formats support
                #      different features to use with this dictionary
                #      specification. For instance the .hocon file format
                #      allows you to specify pieces of strings that come from
                #      environment variables ${LIKE_THIS}, which is very useful
                #      for container oriented processes that want to keep
                #      user names and passwords secret.
                'studio_ml_config': None,

                # Resources needed by experiment, amount of RAM, CPU, GPU, etc
                'resources_needed': {
                    'cpus': 2,
                    'ram': '16g',
                    'hdd': '10g',
                    'gpus': 0,
                    'gpuMem': '5g'
                },
                # Time to allow cs to sleep between checking for results
                'sleep_time': 60,

                # Overwrite for queue name
                # When prefixed with "rmq_", studio uses a RabbitMQ interface
                'queue': None,

                # Whether to turn on debug mode
                # (write submitted candidates to file)
                'debug': False
            },

            # ENN service
            'enn_service_host': 'test.enn.evolution.ml',
            'enn_service_port': 80,

            # Domains can set this as a single string or list of strings
            # to define other.packages that should be searched for
            # domain-specific code
            'extra_packages': None,

            'verbose': False,

            # Tests can set an alternate evaluator to bypass the evaluation
            # When this is set to None, it uses the
            # CompletionServiceEvaluatorSessionTask
            'experiment_host_evaluator': None
        }

        return default_config


    def filter_bucket_names(self, experiment_config):
        """
        Put any user-specified bucket names that are under
        our control through a filter to make them valid
        in a consistent way.
        """

        empty = {}
        completion_service = experiment_config.get('completion_service', empty)
        studio_ml_config = completion_service.get('studio_ml_config', None)

        if studio_ml_config is not None:
            self.filter_one_bucket_name(studio_ml_config, 'database')
            self.filter_one_bucket_name(studio_ml_config, 'storage')

        return experiment_config


    def filter_one_bucket_name(self, studio_ml_config, storage_key):
        """
        :param studio_ml_config: a config dictionary for studio ml
        :param storage_key: an key for storage config information
                where one of the keys is 'bucket'
        """
        empty = {}
        storage_config = studio_ml_config.get(storage_key, empty)
        orig_bucket = storage_config.get('bucket', None)

        if orig_bucket is not None:
            bucket_name_filter = BucketNameFilter()
            use_bucket = bucket_name_filter.filter(orig_bucket)
            if use_bucket != orig_bucket:
                print("Filtering bucket name {0} to {1}".format(
                        orig_bucket, use_bucket))
                storage_config['bucket'] = use_bucket


    def filter_queue_names(self, experiment_config):
        """
        Put any user-specified bucket names that are under
        our control through a filter to make them valid
        in a consistent way.
        """

        empty = {}
        completion_service = experiment_config.get('completion_service', empty)
        orig_queue = completion_service.get('queue', None)

        if orig_queue is not None:
            queue_name_filter = QueueNameFilter()
            use_queue = queue_name_filter.filter(orig_queue)
            if use_queue != orig_queue:
                print("Filtering queue name {0} to {1}".format(
                        orig_queue, use_queue))
                completion_service['queue'] = use_queue

        return experiment_config
