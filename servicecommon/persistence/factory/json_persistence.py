
from servicecommon.persistence.factory.abstract_persistence \
    import AbstractPersistence
from servicecommon.serialization.format.json_serialization_format \
    import JsonSerializationFormat


class JsonPersistence(AbstractPersistence):
    """
    Implementation of the AbstractPersistence class which
    saves JSON data for an object via some persistence mechanism.
    """

    def __init__(self, persistence_mechanism, reference_pruner=None,
                 dictionary_converter=None, pretty=True, logger=None):
        """
        Constructor

        :param persistence_mechanism: the PersistenceMechanism to use
                for storage
        :param reference_pruner: a ReferencePruner implementation
                that knows how to prune/graft repeated references
                throughout the object hierarchy
        :param dictionary_converter: A DictionaryConverter implementation
                that knows how to convert from a dictionary to the object type
                in question.
        :param pretty: a boolean which says whether the JSON is to be
                nicely formatted or not.  indent=4, sort_keys=True
        :param logger: A logger to send messaging to
        """

        super(JsonPersistence, self).__init__(persistence_mechanism)
        self._serialization = JsonSerializationFormat(
                                    reference_pruner=reference_pruner,
                                    dictionary_converter=dictionary_converter,
                                    pretty=pretty,
                                    logger=logger)

    def get_serialization_format(self):
        """
        :return: The SerializationFormat instance to be used in persist()
                 and restore()
        """
        return self._serialization
