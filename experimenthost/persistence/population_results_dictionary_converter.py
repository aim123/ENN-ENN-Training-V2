
from framework.serialization.candidate_dictionary_converter \
    import CandidateDictionaryConverter
from servicecommon.serialization.interface.dictionary_converter \
    import DictionaryConverter


class PopulationResultsDictionaryConverter(DictionaryConverter):
    """
    A DictionaryConverter implementation which knows how to clean up
    PopulationResults for serialization.
    """

    def __init__(self):
        """
        Constructor.
        """
        self.pretty_json_models = True


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

        new_results = []
        candidate_dict_converter = CandidateDictionaryConverter()

        population_results = obj
        for result in population_results:

            new_result = candidate_dict_converter.to_dict(result)
            new_results.append(new_result)

        return new_results


    def from_dict(self, obj_dict):
        """
        :param obj_dict: The data-only dictionary to be converted into an object
        :return: An object instance created from the given dictionary.
                If dictionary is None, the returned object should also be None.
                If obj_dict is None, the returned object should also be None.
                If obj_dict is not the correct type, it is also reasonable
                to return None.
        """
        return obj_dict
