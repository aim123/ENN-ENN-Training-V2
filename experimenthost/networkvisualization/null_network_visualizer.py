# Import base class NetworkVisualizer
from experimenthost.networkvisualization.network_visualizer \
    import NetworkVisualizer

class NullNetworkVisualizer(NetworkVisualizer):
    """
    Null Network Visualizer for visualizing Keras Networks.

    The class overides visualize() method below and returns without
    any further actions or visualization.

    The class uses a default constructor for object initialization.
    """

    def visualize(self, candidate):
        """
        Return from the function without any vizualization
        """

        return
