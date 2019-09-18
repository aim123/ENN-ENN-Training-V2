# Old-school visualizers are lazily imported so we do not
# have to ship them in their quasi-working states.

from experimenthost.networkvisualization.default_keras_network_visualizer \
    import DefaultKerasNetworkVisualizer
from experimenthost.networkvisualization.null_network_visualizer \
    import NullNetworkVisualizer
from experimenthost.networkvisualization.see_nn \
    import SeeNN
from experimenthost.networkvisualization.see_nn_blueprint \
    import SeeNNBlueprint
from experimenthost.networkvisualization.see_nn_module_catalog \
    import SeeNNModuleCatalog


class NetworkVisualizerFactory():
    """
    Factory for NetworkVisualizers.
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


    def create_network_visualizer(self, name, visualizer_config=None):
        """
        Creates a NetworkVisualizer.

        :param name: The string name of the visualizer
        :param visualizer_config: The user-specified configuration dictionary
                for the visualizer. Default is None, meaning use all defaults.
        """

        visualizer = None

        if name is None:
            return visualizer

        lower_name = name.lower()

        # Newer-style visualizers
        if lower_name == "DefaultKerasNetworkVisualizer".lower():
            visualizer = DefaultKerasNetworkVisualizer(self.master_config,
                                        self.data_dict,
                                        self.base_path,
                                        visualizer_config=visualizer_config,
                                        logger=self.logger)

        # Null Network visualizer
        elif lower_name == "NullNetworkVisualizer".lower():
            visualizer = NullNetworkVisualizer()

        # See NN Nested Network visualizer
        elif lower_name == "SeeNN".lower():
            visualizer = SeeNN(self.master_config,
                                        self.data_dict,
                                        self.base_path,
                                        visualizer_config=visualizer_config,
                                        logger=self.logger)
        # See NN Blueprint Network visualizer
        elif lower_name == "SeeNNBlueprint".lower():
            visualizer = SeeNNBlueprint(self.master_config,
                                        self.data_dict,
                                        self.base_path,
                                        visualizer_config=visualizer_config,
                                        logger=self.logger)

        # See NN Module Catalog visualizer
        elif lower_name == "SeeNNModuleCatalog".lower():
            visualizer = SeeNNModuleCatalog(self.master_config,
                                        self.data_dict,
                                        self.base_path,
                                        visualizer_config=visualizer_config,
                                        logger=self.logger)

        # Old-school visualizers are lazily imported so we do not
        # have to ship them in their quasi-working states.
        elif lower_name == "KerasAdvancedSoftOrderNetworkVisualizer".lower():
            from experimenthost.networkvisualization.keras_advanced_soft_order_network_visualizer \
                import KerasAdvancedSoftOrderNetworkVisualizer
            visualizer = KerasAdvancedSoftOrderNetworkVisualizer(self.base_path)

        elif lower_name == "KerasSoftOrderNetworkVisualizer".lower():
            from experimenthost.networkvisualization.keras_soft_order_network_visualizer \
                import KerasSoftOrderNetworkVisualizer
            visualizer = KerasSoftOrderNetworkVisualizer(self.base_path)

        elif lower_name == "KerasCustomNetworkVisualizer".lower():
            from experimenthost.networkvisualization.keras_custom_network_visualizer \
                import KerasCustomNetworkVisualizer
            visualizer = KerasCustomNetworkVisualizer(self.base_path)

        return visualizer
