from datetime import datetime

from framework.util.experiment_filer import ExperimentFiler
from framework.util.generation_filer import GenerationFiler

from servicecommon.persistence.factory.persistence_factory \
    import PersistenceFactory
from servicecommon.persistence.interface.persistence \
    import Persistence
from servicecommon.persistence.mechanism.persistence_mechanisms \
    import PersistenceMechanisms
from servicecommon.serialization.format.serialization_formats \
    import SerializationFormats
from servicecommon.serialization.prep.pass_through_dictionary_converter \
    import PassThroughDictionaryConverter


class ExperimentHostErrorPersistence(Persistence):
    """
    A class which knows how to persist Experiment Host errors to/from file(s)
    when there is an error.

    This class will produce a simple text file of a Traceback.
    The file itself is intended to be human-readable.
    """

    def __init__(self, experiment_dir, generation, timestamp, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param generation: the generation number of the results dict
        :param timestamp: A double timestamp of when the error occurred.
        :param logger: A logger to send messaging to
        """

        filer = ExperimentFiler(experiment_dir)
        error_dir = filer.experiment_file("errors")

        ts_datetime = datetime.fromtimestamp(timestamp)
        time_format = '%Y-%m-%d-%H:%M:%S'
        time_string = ts_datetime.strftime(time_format)

        filer = GenerationFiler(experiment_dir, generation)
        gen_name = filer.get_generation_name()

        basename = "experiment_host_error_{0}_{1}".format(gen_name, time_string)

        dictionary_converter = PassThroughDictionaryConverter()
        factory = PersistenceFactory(object_type="string",
                                     dictionary_converter=dictionary_converter,
                                     logger=logger)
        self.dict_persistence = factory.create_persistence(error_dir, basename,
                        persistence_mechanism=PersistenceMechanisms.LOCAL,
                        serialization_format=SerializationFormats.TEXT,
                        must_exist=False)


    def persist(self, obj):
        """
        Persists the object passed in.

        :param obj: an object to persist
                    In this implementation, we expect a dictionary.
        """

        self.dict_persistence.persist(obj)


    def restore(self):
        """
        :return: an object from some persisted store.
                If the file was not found we return an empty dictionary.
        """

        return self.dict_persistence.restore()
