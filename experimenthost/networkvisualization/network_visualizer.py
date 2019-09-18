class NetworkVisualizer():
    """
    Interface for visualizers of networks.
    """

    def visualize(self, candidate):
        """
        Visualizes the given candidate however the implementation sees fit.

        :param candidate: Dictionary representing the candidate to visualize
        :return: Nothing
        """
        raise NotImplementedError
