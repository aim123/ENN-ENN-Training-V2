
# Lazily imported below to minimize Keras output at startup
#from keras.utils.vis_utils import plot_model

from experimenthost.networkvisualization.abstract_keras_network_visualizer \
    import AbstractKerasNetworkVisualizer


class DefaultKerasNetworkVisualizer(AbstractKerasNetworkVisualizer):
    """
    Default Keras network visualization from keras.plot_model
    """

    def visualize_keras_model(self, keras_model, candidate):
        """
        Subclasses must implement this to visualize the Keras model.

        :param keras_model: The Keras model to visualize
        :param candidate: Dictionary representing the candidate to visualize
        :return: Nothing
        """

        # Lazy import to minimize Keras output at startup
        from keras.utils.vis_utils import model_to_dot
        pydot = model_to_dot(keras_model, show_shapes=True, show_layer_names=True)

        # XXX   Future versions of Keras hint at an expand_nested option here
        #       which will expand nested models.

        self.write_pydot(pydot)
