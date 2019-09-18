
import collections
import copy

from future.utils import iteritems
from past.builtins import basestring

import ruamel.yaml as yaml
from pyhocon import ConfigFactory

class ConfigHandler():
    """
    An abstract class which handles configuration dictionaries
    """

    def import_config(self, config_source, default_config=None):

        # Set up a very basic config dictionary
        config = {}
        if default_config is not None and isinstance(default_config, dict):
            config = copy.deepcopy(default_config)

        # Potentially read config from a file, if config arg is a string filename
        update_source = {}
        if isinstance(config_source, basestring):
            update_source = self.read_config_from_file(config_source)

        # Override entries from the defaults in setupConfig with the
        #     contents of the config arg that was passed in.
        elif isinstance(config_source, dict):
            update_source = config_source

        config = self.deep_update(config, update_source)
        return config


    def deep_update(self, dest, source):
        for key, value in iteritems(source):
            if isinstance(value, collections.Mapping):
                recurse = self.deep_update(dest.get(key, {}), value)
                dest[key] = recurse
            else:
                dest[key] = source[key]
        return dest


    def read_config_from_file(self, filepath):

        # Create a map of our parser methods
        file_extension_to_parser_map = {
            '.conf': 'parse_hocon',
            '.hocon': 'parse_hocon',
            '.json': 'parse_hocon',
            '.properties': 'parse_hocon',
            '.yaml': 'parse_yaml'
        }

        # See what the filepath extension says to use
        parser = None
        for file_extension in list(file_extension_to_parser_map.keys()):
            if filepath.endswith(file_extension):
                parser = file_extension_to_parser_map.get(file_extension)

        if parser is not None:
            config = self.parse_with_method(parser, filepath)
        else:
            print("Could not read {0} as config".format(filepath))
            config = {}

        return config

    def parse_with_method(self, parser, filepath):
        # Python magic to get a handle to the method
        parser_method = getattr(self, parser)

        # Call the parser method with the filepath, get dictionary back
        config = parser_method(filepath)
        return config

    def parse_hocon(self, filepath):
        config = ConfigFactory.parse_file(filepath)
        return config

    def parse_yaml(self, filepath):
        config = {}
        with open(filepath, 'rb') as stream:
            config = yaml.safe_load(stream)
        return config
