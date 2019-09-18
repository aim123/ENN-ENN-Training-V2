
from __future__ import print_function

import os

from past.builtins import basestring

from experimenthost.config.experiment_config import ExperimentConfig
from experimenthost.util.dictionary_overlay import DictionaryOverlay

from framework.domain.config_handler import ConfigHandler
from framework.resolver.domain_resolver import DomainResolver


class MasterConfigReader():
    """
    Reads a root master config file.
    """

    def read_master_config(self, config_file, overlay_config_list=None):
        """
        Loads config file(s).

        This is an outline method, calling smaller specific-purpose methods
        to construct the over-all master configuration for the experiemnt.

        Will examine specific keys in the dictionary to see
        if it needs to load in further config files if necessary.

        :param config_file: a config file reference
        :param overlay_config_list: a list of config files which will be
                    overlayed onto the basis config_file in the order they are
                    listed.
        """

        # Load highest level experiment config, which points to more configs.
        config_path = os.path.abspath(config_file)
        print("Using config_path {0}".format(config_path))

        outside_config = self.read_experiment_config(config_path)
        experiment_config = outside_config.get("experiment_config",
                                               outside_config)

        domain = self.instantiate_domain(experiment_config)

        config_base_dir = self.get_config_base_dir(config_path)
        domain_config = self.read_domain_config(config_base_dir,
                                                outside_config,
                                                experiment_config,
                                                domain)

        builder_config = self.read_builder_config(config_base_dir,
                                                  outside_config,
                                                  experiment_config,
                                                  domain,
                                                  domain_config)

        blueprint_config_path, module_config_path = self.read_algo_configs(
                                                            config_base_dir,
                                                            experiment_config)

        blueprint_config = self.read_blueprint_config(outside_config,
                                                      blueprint_config_path)
        module_config = self.read_module_config(outside_config,
                                                module_config_path)

        # Create a single master config
        master_config = {
            "blueprint_config": blueprint_config,
            "builder_config": builder_config,
            "domain_config": domain_config,
            "experiment_config": experiment_config,
            "module_config": module_config
        }

        master_config = self.do_overlays(master_config, overlay_config_list)

        verbose = experiment_config.get('verbose', False)
        if verbose:
            print(master_config)

        return master_config


    def read_experiment_config(self, config_path):
        """
        Reads the top-level experiment config.
        :param config_path: This is always a file reference
                which can refer to a single-config file with multiple
                sections specified in JSON or YAML, or alternatively,
                a legacy .ini-style experiment file which refers to other
                files of varying formats for different sections of config.
        :return: a single outside config dictionary with the
                'experiment_config' key filled in with what was read in
                from the file.
        """

        exp_config_builder = ExperimentConfig()
        outside_config = exp_config_builder.build_config(config_path)

        return outside_config


    def instantiate_domain(self, experiment_config):
        """
        Resolve and load code for the DomainConfig class.
        The DomainConfig class is only used for doing configuration operations

        :param experiment_config: The Experiment Config dictionary which
            contains a reference to the domain name.
        :return:  An instantiation of the DomainConfig class,
                   loaded from the various references in the experiment config.
        """

        domain_resolver = DomainResolver()

        domain_name = experiment_config.get('domain')
        class_name = experiment_config.get('domain_config_class_name')
        extra_packages = experiment_config.get('extra_packages')
        verbose = experiment_config.get('verbose')

        domain_class = domain_resolver.resolve(domain_name,
                                class_name=class_name,
                                extra_packages=extra_packages,
                                verbose=verbose)
        domain = domain_class()

        return domain


    def get_config_base_dir(self, config_path):
        """
        Used for legacy multi-file config references
        :param config_path: The path to the root configuration file.
        :return: The directory in which references to other config files
                    can be found.
        """

        config_base_dir = None
        if config_path is not None:
            config_base_dir = os.path.dirname(config_path)

        return config_base_dir


    def read_domain_config(self, config_base_dir, outside_config,
                           experiment_config, domain):
        """
        :param config_base_dir: Basis directory for config file references
        :param outside_config: The outermost config dictionary read in
                    as the root config reference.
        :param experiment_config: The experiment_config dictionary reference
                    within the outside_config dictionary (used for
                    legacy compatibility)
        :param domain: The DomainConfig instance
        :return: The domain_config dictionary to be used, potentially read from
                    various sources.
        """

        verbose = experiment_config.get('verbose', False)

        # Read in potential
        domain_config_in = outside_config.get('domain_config', None)
        if domain_config_in is None:
            domain_config_in = experiment_config.get('domain_config_in', None)

        # If what we got is already a dictionary, then we are done
        domain_config = None
        if isinstance(domain_config_in, dict):
            domain_config = domain.build_config(domain_config_in,
                                            verbose=verbose)

        elif isinstance(domain_config_in, basestring):
            if config_base_dir is not None:

                domain_path = os.path.join(config_base_dir, domain_config_in)
                domain_config = domain.build_config(domain_path)

            else:
                domain_config = domain.build_config(domain_config_in)

        return domain_config


    def read_builder_config(self, config_base_dir, outside_config,
                            experiment_config, domain, domain_config):
        """
        :param config_base_dir: Basis directory for config file references
        :param outside_config: The outermost config dictionary read in
                    as the root config reference.
        :param experiment_config: The experiment_config dictionary reference
                    within the outside_config dictionary (used for
                    legacy compatibility)
        :param domain: The DomainConfig instance
        :param domain_config: The domain_config previously constructed.
        :return: The builder_config dictionary to be used, potentially read from
                    various sources, and augmented by various method on
                    the DomainConfig object.
        """

        network_builder_config_in = \
                      outside_config.get('builder_config', None)
        if network_builder_config_in is None:
            network_builder_config_in = \
                        experiment_config.get('network_builder_config_in', None)

        # If what we got is already a dictionary, then we are done
        builder_config = None
        if isinstance(network_builder_config_in, dict):
            builder_config = network_builder_config_in

        elif isinstance(network_builder_config_in, basestring):

            use_builder_config = network_builder_config_in
            if config_base_dir is not None:
                use_builder_config = \
                    os.path.join(config_base_dir, network_builder_config_in)

            # See if the domain adds on some input/output shape-type information
            # for builder
            config_handler = ConfigHandler()
            builder_config = config_handler.import_config(use_builder_config)

        builder_config = self.add_builder_config_add_ons(builder_config,
                                            domain, domain_config)

        return builder_config


    def read_algo_configs(self, config_base_dir, experiment_config):
        """
        :param config_base_dir: Basis directory for config file references
        :param experiment_config: The experiment_config dictionary reference
                    within the outside_config dictionary (used for
                    legacy compatibility)
        :return: A tuple of references to the blueprint and module config
                files (legacy mechanism).  Tuple of (None, None) is returned
                if these are not specified (expected for newer, single-config)
        """

        algo_configs_in = \
            experiment_config.get('evolution_algorithm_config_in', None)

        # Allow for new specification
        if algo_configs_in is None:
            algo_configs = (None, None)
            return algo_configs

        algo_configs = algo_configs_in
        if config_base_dir is not None:
            algo_configs = tuple(os.path.join(config_base_dir, x) \
                                    for x in algo_configs_in)

        return algo_configs


    def read_blueprint_config(self, outside_config,
                              blueprint_config_path):
        """
        Reads in the blueprint_config to be used.

        :param outside_config: The outermost config dictionary read in
                    as the root config reference.
        :param blueprint_config_path: The path to the blueprint config file
                    (if any).
        :return: The blueprint_config dictionary to be used, potentially
                    read from various sources.
        """


        blueprint_config = None

        blueprint_config_in = blueprint_config_path
        if blueprint_config_in is None:
            blueprint_config_in = outside_config.get('blueprint_config', None)

        if blueprint_config_in is not None:
            if isinstance(blueprint_config_in, dict):
                blueprint_config = blueprint_config_in
            elif isinstance(blueprint_config_in, basestring):
                config_handler = ConfigHandler()
                blueprint_config = config_handler.import_config(
                                                        blueprint_config_in)

        return blueprint_config


    def read_module_config(self, outside_config, module_config_path):
        """
        Reads in the module_config to be used.

        :param outside_config: The outermost config dictionary read in
                    as the root config reference.
        :param module_config_path: The path to the module config file (if any).
        :return: The module_config dictionary to be used, potentially read from
                    various sources.
        """

        module_config = None

        module_config_in = module_config_path
        if module_config_in is None:
            module_config_in = outside_config.get('module_config', None)

        if module_config_in is not None:
            if isinstance(module_config_in, dict):
                module_config = module_config_in
            elif isinstance(module_config_in, basestring):
                config_handler = ConfigHandler()
                module_config = config_handler.import_config(module_config_in)

        return module_config


    def add_builder_config_add_ons(self, builder_config, domain,
                                    domain_config):
        """
        Uses the DomainConfig object to potentially modify
        the builder_config object that was read in with
        domain-specific addenda, so it can be sent to the service

        :param builder_config: The basis builder_config dictionary
                    read in from files.
        :param domain: The DomainConfig instance
        :param domain_config: The domain_config previously constructed.
        :return: The builder_config dictionary to be used,
                    augmented by various method on the DomainConfig object.
        """

        domain_specific_dict = domain_config.get('info', {})

        builder_add_ons = domain.generate_builder_config_add_ons(
                                    domain_specific_dict)

        # Create a composite builder config where the builder_add_ons
        # are added to what is in the builder_config, but where the
        # builder_config itself can override. The idea here is to future-proof
        # for when moving the config relating to the builder is removed from
        # the domain config, and everything is specified directly in the
        # builder_config.
        composite_builder_config = {}
        composite_builder_config.update(builder_add_ons)
        if builder_config is not None:
            composite_builder_config.update(builder_config)

        return composite_builder_config


    def do_overlays(self, basis, overlay_list):
        """
        :param basis: The basis config
        :param overlay_list: A list of overlay config file references to apply
                        to the basis, in order.
        :return: A new config with all the overlays applied (if any)
        """

        if overlay_list is None:
            print("No overlays applied")
            return basis

        if not isinstance(overlay_list, list):
            overlay_list = [overlay_list]

        overlayer = DictionaryOverlay()
        config_handler = ConfigHandler()

        # Loop through each of the overlay config references,
        # reading in each overlay config and applying it as an overlay
        # to the previous basis dictionary.
        result = basis
        for overlay_config in overlay_list:
            print("Applying overlay {0}".format(overlay_config))
            overlay = config_handler.import_config(overlay_config)
            result = overlayer.overlay(result, overlay)

        return result
