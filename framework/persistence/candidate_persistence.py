
from framework.serialization.candidate_dictionary_converter \
    import CandidateDictionaryConverter
from framework.util.generation_filer import GenerationFiler

from servicecommon.persistence.easy.easy_json_persistence \
    import EasyJsonPersistence


class CandidatePersistence(EasyJsonPersistence):
    """
    A class which knows how to persist a candidate dict to/from file(s).

    A candidate contains a few major fields:
        * id - the string id of the candidate unique to at least the experiment
        * identity -  a dictionary containing information about the
                        birth circumstances of the candidate
        * interpretation - which contains a digestible form of the evolved
                            genetic material
        * metrics - a dictionary containing measurements from evaluation of the
                    candidate

    This class will produce a pretty JSON file that can be used to
    produce/recover Candidates from a generation directory (if a generation
    is given) or an experiment results directory (if a generation is not given).

    The file itself is intended to be human-readable as well as
    machine-readable.
    """

    def __init__(self, experiment_dir, candidate_id=None, generation=None,
                    base_name=None, logger=None):
        """
        Constructor.

        :param experiment_dir: the directory where experiment results go
        :param candidate_id: the id of the candidate
        :param generation: the generation number for the candidate
        :param base_name: a full base name to use (minus extension)
        :param logger: The logger to use for messaging
        """

        use_base_name = base_name
        if use_base_name is None:
            use_base_name = self.get_base_name(candidate_id, generation)

        use_dir = experiment_dir
        if generation is not None:
            filer = GenerationFiler(experiment_dir, generation)
            use_dir = filer.get_generation_dir()

        dictionary_converter = CandidateDictionaryConverter(
                                        allow_restore_none=False)
        super(CandidatePersistence, self).__init__(
                            base_name=use_base_name,
                            folder=use_dir,
                            dictionary_converter=dictionary_converter,
                            must_exist=True,
                            logger=logger)

    def get_base_name(self, candidate_id, generation):
        """
        :param candidate_id: the id of the candidate
        :param generation: the generation number for the candidate
        :return: the base name for the file to be persisted
        """
        base_name = "candidate_{0}".format(candidate_id)

        # Note: _ is pythonic for unused variable
        _ = generation

        return base_name
