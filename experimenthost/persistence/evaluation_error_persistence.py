from datetime import datetime

from framework.serialization.candidate_dictionary_converter \
    import CandidateDictionaryConverter
from framework.util.experiment_filer import ExperimentFiler
from framework.util.generation_filer import GenerationFiler

from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence



class EvaluationErrorPersistence(EasyJsonPersistence):
    """
    A class which knows how to persist a worker response dict to/from file(s)
    when there is an error.

    The Worker Response is a dictionary of results coming back from
    a Studio ML worker.  We only save these if there is an error
    coming from the worker.

    This class will produce a pretty JSON file that can be used to
    produce Worker Response Dictionaries from a generation directory.
    The file itself is intended to be human-readable as well as
    machine-readable.
    """

    def __init__(self, experiment_dir, generation, candidate_id,
                    timestamp, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param generation: the generation number of the results dict
        :param candidate_id: The id of the candidate that had the error
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

        base_name = "evaluation_error_{0}_candidate_{1}_{2}".format(
                                gen_name, candidate_id, time_string)

        super(EvaluationErrorPersistence, self).__init__(base_name=base_name,
                        folder=error_dir,
                        dictionary_converter=CandidateDictionaryConverter(),
                        logger=logger)
