
from experimenthost.tasks.session_task import SessionTask


class ListCandidateIdsSessionTask(SessionTask):
    """
    SessionTask that lists all available candidate ids from a checkpoint.
    """

    def run(self):
        """
        Entry point for the session task execution to take over.
        """
        print("Running ListCandidateIdsSessionTask")

        population_response = self.session.get_population(
            self.experiment_dir,
            self.checkpoint_id)

        if population_response is None:
            print("No checkpoint {0} found for experiment {1}".format(
                self.checkpoint_id,
                self.experiment_dir))
            return

        population = population_response.get("population", None)
        generation = population_response.get("generation_count", -1)

        print("Ids from gen {0} from experiment {1} checkpoint {2} :".format(
            generation,
            self.experiment_dir,
            self.checkpoint_id))

        for candidate in population:
            candidate_id = str(candidate.get('id', None))
            print(candidate_id)
