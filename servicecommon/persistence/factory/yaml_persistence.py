
from servicecommon.persistence.factory.abstract_persistence \
    import AbstractPersistence
from servicecommon.serialization.format.yaml_serialization_format \
    import YamlSerializationFormat


class YamlPersistence(AbstractPersistence):
    """
    Implementation of the AbstractPersistence class which
    saves YAML data for an object via some persistence mechanism.
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
        :param pretty: a boolean which says whether the YAML is to be
                nicely formatted or not.  indent=4, sort_keys=True
        :param logger: a logger to send messaging to
        """

        super(YamlPersistence, self).__init__(persistence_mechanism)
        self._serialization = YamlSerializationFormat(
                                    reference_pruner=reference_pruner,
                                    dictionary_converter=dictionary_converter,
                                    pretty=pretty,
                                    logger=logger)

    def get_serialization_format(self):
        return self._serialization
