
from framework.persistence.candidate_persistence import CandidatePersistence


class BestFitnessCandidatePersistence(CandidatePersistence):
    """
    A class which knows how to persist a given candidate as a best candidate.
    (That the candidate is "best" contributes to the naming of the file)
    """

    def get_base_name(self, candidate_id, generation):
        """
        :param candidate_id: the id of the candidate
        :param generation: the generation number for the candidate
        :return: the base name for the file to be persisted
        """

        # XXX For now, just using fitness.
        #     Later on, can incorporate multi-objective goals.
        metric_name = "fitness"

        base_name = "best_{0}_candidate".format(metric_name)
        if generation is None:
            # Only put the candidate id on if we do not have the generation
            base_name = "{0}_id-{1}".format(base_name, candidate_id)

        return base_name
