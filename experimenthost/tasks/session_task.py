
import logging


class SessionTask():
    """
    Superclass of tasks performed by the SessionServer.
    """

    def __init__(self, session, master_config, experiment_dir,
                 fitness_objectives, checkpoint_id=None):
        """
        Constructor.

        :param session: The session with which the task can communicate
                    with the service
        :param master_config: The master config for the task
        :param experiment_dir: The experiment directory for results
        :param fitness_objectives: The FitnessObjectives object
        :param checkpoint_id: The checkpoint id (if any) relevant to the task.
        """
        self.session = session
        self.master_config = master_config
        self.experiment_dir = experiment_dir
        self.fitness_objectives = fitness_objectives
        self.checkpoint_id = checkpoint_id
        self.logger = logging.getLogger('EnnServiceSession')


    def run(self):
        """
        Entry point for the session task execution to take over.
        """
        raise NotImplementedError
