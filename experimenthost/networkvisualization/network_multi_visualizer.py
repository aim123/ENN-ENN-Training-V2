from experimenthost.networkvisualization.network_visualizer \
    import NetworkVisualizer
from experimenthost.networkvisualization.network_visualizer_factory \
    import NetworkVisualizerFactory

from experimenthost.util.candidate_util import CandidateUtil

from servicecommon.parsers.canonical_multi_config_parser \
    import CanonicalMultiConfigParser


class NetworkMultiVisualizer(NetworkVisualizer):
    """
    Class to aid in handling the configuration and visualize() calling
    for potentially multiple Network Visualizers.

    While this class is itself a NetworkVisualizer, we do not enter
    this in the NetworkVisualizerFactory to avoid a dependency tangle.
    """

    def __init__(self, master_config, data_dict, base_path, logger=None):
        """
        Constructor.

        :param master_config: The master config for the experiment
                from which all other sub-configs can be obtained.
        :param data_dict: The data dictionary used in the evaluator.
                This is often needed by domains in order that the model
                is built with the correct dimensionality
        :param base_path: The base pathname to which the visualization will
                be saved.  It is up to the implementation to add any file
                suffixes or further identifiers onto this path.
        :param logger: A logger to send messaging to
        """
        self.master_config = master_config
        self.data_dict = data_dict
        self.base_path = base_path
        self.logger = logger


    def visualize(self, candidate):
        """
        Visualizes the given candidate using one or more
        NetworkVisualizers available from the Factory.

        :param candidate: Dictionary representing the candidate to visualize
        :return: Nothing
        """

        candidate_util = CandidateUtil()
        candidate_id = candidate_util.get_candidate_id(candidate)

        # Get the value for the key describing how network visualization
        # is to be performed.
        experiment_config = self.master_config.get("experiment_config", {})
        vis_value = experiment_config.get("network_visualization", None)

        # Parse the value to be in a cannonical form of a list of
        # configurations for visualizers
        name_key = "name"
        parser = CanonicalMultiConfigParser(name_key=name_key,
                                            logger=self.logger)
        vis_config_list = parser.parse(vis_value)

        vis_factory = NetworkVisualizerFactory(self.master_config,
                                               self.data_dict,
                                               self.base_path,
                                               logger=self.logger)

        # Loop through the compiled vis_config_list to invoke all the desired
        # NetworkVisualizers.
        for vis_config in vis_config_list:

            # Get the name to use for the factory from the config
            vis_name = vis_config.get(name_key, None)

            # Create the visualizer
            visualizer = vis_factory.create_network_visualizer(vis_name,
                                                               vis_config)

            if visualizer is not None:
                # We have a visualizer. Draw!
                print("Using {0} to draw candidate {1}".format(vis_name,
                                                            candidate_id))
                visualizer.visualize(candidate)

            else:
                # Do not fail just because of a typo.
                print("Don't know network visualizer '{0}'. Skipping.".format(
                                                            vis_name))
