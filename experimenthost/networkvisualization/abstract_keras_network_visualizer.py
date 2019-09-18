from experimenthost.networkvisualization.network_visualizer \
    import NetworkVisualizer
from experimenthost.persistence.network_visualizer_persistence \
    import NetworkVisualizerPersistence
from experimenthost.util.dictionary_overlay import DictionaryOverlay
from framework.resolver.evaluator_resolver import EvaluatorResolver


class AbstractKerasNetworkVisualizer(NetworkVisualizer):
    """
    Abstract NetworkVisualizer for visualizing Keras networks.

    This class has some base methods for translating a candidate from the
    ENN Service into a Keras model.  This requires an instance of the domain's
    NetworkEvaluator class.

    Subclasses override the visualize_keras_model() method below.
    """

    def __init__(self, master_config, data_dict, base_path,
                    suffix="", visualizer_config=None, logger=None):
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
        :param suffix: A string suffix, potentially supplied by subclasses
                which is added to the base_path to distinguish one visualizer's
                output from another when multiple visualizers are configured.
        :param visualizer_config: The user-specified configuration dictionary
                for the visualizer. Default is None, meaning use all defaults.
        :param logger: A logger to send messaging to
        """
        self.master_config = master_config
        self.data_dict = data_dict
        self.base_path = base_path
        self.suffix = suffix
        self.logger = logger

        # Use a common policy for all subclasses to specify their
        # default configuration and methodology for overlaying
        # user-specified config (passed in as arg).
        self.visualizer_config = self.get_default_visualizer_config()
        if visualizer_config is not None and \
            isinstance(visualizer_config, dict):

            # Do a deep overlay
            overlayer = DictionaryOverlay()
            self.visualizer_config = overlayer.overlay(self.visualizer_config,
                                                        visualizer_config)


    def visualize(self, candidate):
        """
        Visualizes the given candidate however the implementation sees fit.

        :param candidate: Dictionary representing the candidate to visualize
        :return: Nothing
        """
        keras_model = self.interpret_keras_model(candidate)

        if keras_model is None:
            print("Could not visualize Keras JSON")
            return

        self.visualize_keras_model(keras_model, candidate)


    def visualize_keras_model(self, keras_model, candidate):
        """
        Subclasses must implement this to visualize the Keras model.

        :param keras_model: The Keras model to visualize
        :param candidate: Dictionary representing the candidate to visualize
        :return: Nothing
        """
        raise NotImplementedError


    def interpret_keras_model(self, candidate):
        """
        Interprets the candidate into a Keras model that can be visualized

        :param candidate: The candidate dictionary from the ENN Service
        :return: a Keras model
        """

        # Get the NetworkEvaluator so we can build the network
        evaluator = self.resolve_evaluator()
        if evaluator is None:
            return None

        empty_dict = {}

        # Get all the arguments we need to build the network from the candidate
        interpretation = candidate.get("interpretation", empty_dict)
        model_json_string = interpretation.get("model", None)
        global_hyperparameters = interpretation.get("global_hyperparameters",
                                                    None)
        candidate_id = candidate.get("id", None)
        domain_config = self.master_config.get("domain_config", empty_dict)

        model = evaluator.build_training_model(candidate_id,
                                model_json_string,
                                global_hyperparameters,
                                domain_config,
                                self.data_dict)

        return model


    def resolve_evaluator(self):
        """
        Resolve and load code for the evaluator class
        Note we do not actually use the reference here, but it's better
        to find problems before sending things out for distribution.

        :return: An instantiation of the ModelEvaluator class,
                loaded from the various references in the experiment
                and domain config.
        """

        experiment_config = self.master_config.get('experiment_config')
        domain_config = self.master_config.get('domain_config')

        evaluator_resolver = EvaluatorResolver()
        evaluator_class = evaluator_resolver.resolve(
                experiment_config.get('domain'),
                class_name=domain_config.get('evaluator_class_name', None),
                evaluator_name=experiment_config.get('network_builder'),
                extra_packages=experiment_config.get('extra_packages'),
                verbose=experiment_config.get('verbose'))

        evaluator_instance = None
        if evaluator_class is not None:
            evaluator_instance = evaluator_class()

        return evaluator_instance


    def write_pydot(self, pydot):
        """
        This function writes out the given pydot graph to an image file
        specified by the "format" key of the visualizer_config
        and the base_path specified in the constructor.

        This method uses the Persistence infrastructure to save via the
        appropriate persistence mechanism(s).

        :param pydot: The pydot graph compiled by the subclass implementation.
        :return: Nothing
        """

        image_bytes = None

        image_format = self.visualizer_config.get("format", "png")

        if image_format == 'png':
            # Conver dot file to PNG byte stream
            image_bytes = pydot.create_png()

        elif image_format == 'pdf':
            # Conver dot file to PNG byte stream
            image_bytes = pydot.create_pdf()

        use_base = self.base_path + self.suffix
        self.write_image_bytes(image_bytes, use_base, image_format)


    def write_image_bytes(self, image_buffer, image_base_name, image_format):
        """
        Writes out the raw bytes of an image_buffer to a file
        with the given image_base_name and the image_format_extension.
        This method will do common handling of just how the file
        is to be persisted (think local file vs S3).

        :param image_buffer: The bytes, bytearray or IOBytes containing
                the formatted image data in its entirety that is to be
                written out to the file.
        :param image_base_name: The base name for the image
        :param image_format: the format of the image to be used as file
                extension ('png' or 'pdf', for instance)
        :return: Nothing
        """

        # Write out the byte stream to a png image
        if image_buffer is not None:
            persistence = NetworkVisualizerPersistence(image_base_name,
                                                        image_format,
                                                        logger=self.logger)
            persistence.persist(image_buffer)


    def get_default_visualizer_config(self):
        """
        A chance for subclasses to supply a default configuration
        on top of which any user mods are made.

        :return: a dictionary populated with the default configuration
                for the visualizer.
        """

        default_config = {
            "format": "png"
        }
        return default_config
