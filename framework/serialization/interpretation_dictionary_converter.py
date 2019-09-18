
import copy
import json

from past.builtins import basestring

from servicecommon.serialization.interface.dictionary_converter \
    import DictionaryConverter


class InterpretationDictionaryConverter(DictionaryConverter):
    """
    A DictionaryConverter implementation that converts a
    Candidate's Interpretation back and forth from a dictionary.

    The "model" field in particular comes back from the service as
    a single big JSON string.  This implementation detects such a string
    and converts it to a dictionary for the purposes of easier examination.
    """

    def to_dict(self, obj):
        """
        :param obj: The object to be converted into a dictionary
        :return: A data-only dictionary that represents all the data for
                the given object, either in primitives
                (booleans, ints, floats, strings), arrays, or dictionaries.
                If obj is None, then the returned dictionary should also be
                None.  If obj is not the correct type, it is also reasonable
                to return None.
        """

        # The object itself is already a dictionary, it's just that some
        # fields which should also be dictionaries are sometimes not.
        if not isinstance(obj, dict):
            return None

        interpretation = obj
        obj_dict = copy.deepcopy(interpretation)

        model_json = obj_dict.get("model", None)
        if model_json is not None:
            if isinstance(model_json, basestring):
                model_dict = json.loads(model_json)
                obj_dict["model"] = model_dict

        return obj_dict


    def from_dict(self, obj_dict):
        """
        :param obj_dict: The data-only dictionary to be converted into an object
        :return: An object instance created from the given dictionary.

                If dictionary is None, then the returned object should also be
                None.

                If obj_dict is None, then the returned object should also be
                None.

                If obj_dict is not the correct type, it is also reasonable
                to return None.
        """
        # The object itself is already a dictionary, it's just that some
        # fields which should also be dictionaries are sometimes not.
        if obj_dict is None or \
            not isinstance(obj_dict, dict):
            return None

        use_obj_dict = obj_dict

        model_dict = obj_dict.get("model", None)
        if model_dict is not None:
            if isinstance(model_dict, dict):
                use_obj_dict = copy.deepcopy(obj_dict)
                model_json = json.dumps(model_dict)
                use_obj_dict["model"] = model_json

        return use_obj_dict
