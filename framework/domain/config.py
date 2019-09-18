
import copy

from framework.domain.config_handler import ConfigHandler


class Config():
    """
    Domain Configuration object base class
    """

    def read_config(self, dict_reference, default_config=None,
                    extra_config_key=None, verbose=False):
        """
        Reads a config given a reference to a dictionary.

        :param dict_reference: A reference to a dictionary.
                Can be an existing dictionary or a filename
                to be read in.
        :param default_config: Default None.  When specified,
                this dictionary contains default key/value pairs
                which are to be used when the keys are absent from
                the given dict_reference
        :param extra_config_key: Default None.
                When specified, keys that exist in the dict_reference, but that
                are not in the default_config dictionary will be siphoned
                off to a sub-dictionary whose key is the string value.
        :param verbose: Controls how chatty the process is. Default False.
        """

        fully_specified_config = {}
        if default_config is not None:
            fully_specified_config = copy.deepcopy(default_config)

        # This is where the keys that were not in the default config go
        if extra_config_key is not None:
            fully_specified_config[extra_config_key] = {}

        if dict_reference is None:
            return fully_specified_config

        # Read config in from file
        handler = ConfigHandler()
        update_config = handler.import_config(dict_reference)

        # Do the separation into extra_config for non-defaults
        fully_specified_config = self.update(fully_specified_config,
                                             update_config, extra_config_key,
                                             verbose)

        return fully_specified_config


    def update(self, fully_specified_config, overlay_config,
               extra_config_key=None, verbose=False):
        """
        Updates a fully-specified dictionary with a dictionary of
        key/value pair overlays to supercede the values in the fully-specified
        dictionary with defaults.

        :param fully_specified_config: A dictionary assumed to be fully
                specified having all keys to be used associated with their
                default values.
        :param overlay_config: A dictionary of partially specified key/value
                pairs to be overlayed on top of the fully_specified_config.
        :param extra_config_key: Default None.
                When specified, keys that exist in the dict_reference, but that
                are not in the default_config dictionary will be siphoned
                off to a sub-dictionary whose key is the string value.
        :param verbose: Controls how chatty the process is. Default False.
        :return: a modified version of the fully_specified_config
        """
        for key, value in list(overlay_config.items()):
            self._update_parameter(fully_specified_config,
                                   key, value,
                                   extra_config_key,
                                   verbose)
        return fully_specified_config


    def _update_parameter(self, fully_specified_config, key, value,
                          extra_config_key=None, verbose=False):
        """
        :param fully_specified_config: A dictionary assumed to be fully
                specified having all keys to be used associated with their
                default values.
        :param key: a single key to update in the fully_specified_config
        :param value: value to update the single key in the
                fully_specified_config
        :param extra_config_key: Default None.
                When specified, keys that exist in the dict_reference, but that
                are not in the default_config dictionary will be siphoned
                off to a sub-dictionary whose key is the string value.
        :param verbose: Controls how chatty the process is. Default False.
        :return: a modified version of the fully_specified_config
        """
        if key in fully_specified_config or extra_config_key is None:
            if isinstance(value, dict):
                full_sub = fully_specified_config.get(key, None)
                if full_sub is None:
                    # We have a dict, but no key was there before
                    # Just assign it.
                    fully_specified_config[key] = value
                elif isinstance(full_sub, dict):
                    # Recursively update the dict
                    self.update(full_sub, value, extra_config_key=None,
                                verbose=verbose)
                else:
                    fully_specified_config[key] = value
            else:
                fully_specified_config[key] = value
        else:
            fully_specified_config[extra_config_key][key] = value
            if verbose:
                print("additional parameter found [{0} = {1}]".format(
                        key, str(value)))

        return fully_specified_config


    def key_prefix_to_subsection(self, source_dict, key_prefix, subsection_key):
        """
        Migrate a group of keys in a source dictionary with a common prefix
        to its own sub-dictionary.
        :param source_dict: The source dictionary to examine
        :param key_prefix: The key prefix which connotes a move to the
                new sub-dictionary
        :param subsection_key: The dictionary key for the new sub-dictionary
        :return: A copy of source_dict with corresponding keys moved
            to the new subsection. If no keys match the key_prefix, then
            no new sub-dictionary will be created.
        """


        # Don't modify any arguments
        dest = copy.deepcopy(source_dict)

        # Look for some special cases
        if key_prefix is None:
            return dest

        key_prefix_len = len(key_prefix)
        if key_prefix_len == 0:
            return dest

        sub_dict = dest.get(subsection_key, None)

        # Examine each key
        # Use the source to avoid changing the dict out from under
        # the key iteration.
        for key in source_dict:

            # Don't bother with the key if it doesn't match the prefix
            if not key.startswith(key_prefix):
                continue

            # Don't mess with the sub-dict itself
            if key == subsection_key:
                continue

            # We found a key. Create the sub-dict if necessary
            if sub_dict is None:
                dest[subsection_key] = {}
                sub_dict = dest[subsection_key]

            # Find the new key for the sub-dict
            new_key = key[key_prefix_len:]
            new_key = self.remove_leading_underscores(new_key)

            # Remove the old key from the dictionary and
            # put the old value in the sub-dict under the new key
            value = dest.pop(key)
            sub_dict[new_key] = value

        return dest


    def remove_leading_underscores(self, instring):
        """
        :param instring: a string
        :return: a new string with any leading underscores removed
        """
        outstring = instring

        while outstring.startswith('_'):
            outstring = outstring[1:]

        return outstring
