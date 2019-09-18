
import copy

from framework.serialization.metrics_dictionary_converter \
    import MetricsDictionaryConverter
from framework.serialization.interpretation_dictionary_converter \
    import InterpretationDictionaryConverter

from servicecommon.serialization.prep.pass_through_dictionary_converter \
    import PassThroughDictionaryConverter


class CandidateDictionaryConverter(PassThroughDictionaryConverter):
    """
    A DictionaryConverter implementation which knows how to clean up
    a Candidate dictionary for serialization.
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

        # Our object is a Candidate
        candidate = obj

        # Make a deep copy of the original object so we can
        # modify at will without messing with the argument contents.
        persisted_candidate = copy.deepcopy(candidate)

        # Clean up the metrics
        empty = {}
        metrics = candidate.get('metrics', empty)
        metrics_dict_converter = MetricsDictionaryConverter()
        clean_metrics = metrics_dict_converter.to_dict(metrics)

        # The model itself can be a JSON string which is not so pretty
        # If that is the case, convert that guy into a dictionary so
        # it is easier to read by humans.
        old_interpretation = candidate.get('interpretation', empty)
        interp_dict_converter = InterpretationDictionaryConverter()
        persisted_interpretation = interp_dict_converter.to_dict(
                                            old_interpretation)

        if persisted_candidate is None:
            persisted_candidate = {}

        persisted_candidate['interpretation'] = persisted_interpretation
        persisted_candidate['metrics'] = clean_metrics

        return persisted_candidate


    def from_dict(self, obj_dict):
        """
        :param obj_dict: The data-only dictionary to be converted into an object
        :return: An object instance created from the given dictionary.
                If dictionary is None, the returned object should also be None.
                If obj_dict is None, the returned object should also be None.
                If obj_dict is not the correct type, it is also reasonable
                to return None.
        """

        if obj_dict is None or \
            not isinstance(obj_dict, dict):

            # Consult the superclass as to the state of allow_restore_none
            no_dict = super(CandidateDictionaryConverter, self).from_dict(
                                            obj_dict)
            return no_dict

        persisted_candidate = obj_dict
        candidate = copy.deepcopy(persisted_candidate)

        # The only thing we want to be sure of is that when we read the
        # candidate back in that we are getting things back the way the rest of
        # the system expects.
        #
        # The metrics have been cleaned, but we do not have enough information
        # to restore those to any numpy classes. Oh well.
        #
        # The model JSON, however should be restored back to its string state,
        # at least until we can get the rest of the system to not expect a
        # JSON string.

        empty = {}
        persisted_interpretation = persisted_candidate.get(
                                        'interpretation', empty)
        interp_dict_converter = InterpretationDictionaryConverter()
        interpretation = interp_dict_converter.from_dict(
                                            persisted_interpretation)
        candidate['interpretation'] = interpretation

        return candidate
