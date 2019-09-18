
from servicecommon.persistence.factory.abstract_persistence \
    import AbstractPersistence
from servicecommon.serialization.format.chained_serialization_format \
    import ChainedSerializationFormat
from servicecommon.serialization.format.buffered_gzip_serialization_format \
    import BufferedGzipSerializationFormat
from servicecommon.serialization.format.json_serialization_format \
    import JsonSerializationFormat


class JsonGzipPersistence(AbstractPersistence):
    """
    Implementation of the AbstractPersistence class which
    saves gzipped JSON data for an object via some persistence mechanism.
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
        :param logger: a logger to send messaging to
        """

        super(JsonGzipPersistence, self).__init__(persistence_mechanism)
        chained = ChainedSerializationFormat()
        chained.add_serialization_format(JsonSerializationFormat(
                                    reference_pruner=reference_pruner,
                                    dictionary_converter=dictionary_converter,
                                    pretty=pretty,
                                    logger=logger))
        chained.add_serialization_format(BufferedGzipSerializationFormat(
                                    persistence_mechanism.folder,
                                    persistence_mechanism.base_name))
        self._serialization = chained


    def get_serialization_format(self):
        return self._serialization
