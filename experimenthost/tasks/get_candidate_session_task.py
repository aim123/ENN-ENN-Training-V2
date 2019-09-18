
from experimenthost.tasks.session_task import SessionTask


class GetCandidateSessionTask(SessionTask):
    """
    SessionTask which prints information about a specific candidate.
    """

    def __init__(self, session, master_config, experiment_dir,
                 fitness_objectives, checkpoint_id=None, get_candidate=None):
        """
        Constructor.

        :param session: The session with which the task can communicate
                    with the service
        :param master_config: The master config for the task
        :param experiment_dir: The experiment directory for results
        :param fitness_objectives: The FitnessObjectives object
        :param checkpoint_id: The checkpoint id (if any) relevant to the task.
        :param get_candidate: The candidate id to get information about
        """
        super(GetCandidateSessionTask, self).__init__(session,
                                                      master_config,
                                                      experiment_dir,
                                                      fitness_objectives,
                                                      checkpoint_id)
        self.get_candidate = get_candidate

    def run(self):
        """
        Entry point for the session task execution to take over.
        """
        print("Running GetCandidateSessionTask for {0}".format(
            self.get_candidate))

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

        found = None
        for candidate in population:
            candidate_id = str(candidate.get('id', None))
            if candidate_id == self.get_candidate:
                found = candidate
                break

        if found is None:
            print("No candidate {0} found in checkpoint {1} for exp {2}".format(
                        self.get_candidate,
                        self.checkpoint_id,
                        self.experiment_dir))
            return

        self.act_on_candidate(found, generation)

    def act_on_candidate(self, candidate, generation):
        """
        :param candidate: The candidate dictionary we want to do something with
        :param generation: the generation number the candidate belongs to
        """
        # Note: _ is pythonic for an unused variable
        _ = generation
        print(str(candidate))
