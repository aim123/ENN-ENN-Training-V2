import json
import pydot

# Import Other classes
from experimenthost.networkvisualization.visualizer_utils \
    import VisualizerUtils

from experimenthost.networkvisualization.graph_utils \
        import GraphUtils

from experimenthost.networkvisualization.visualizer_shared_functions \
        import VisualizerSharedFunctions

# Import Base class
from experimenthost.networkvisualization.see_nn \
        import SeeNN

class SeeNNModuleCatalog(SeeNN):
    """
    This class inherits from AbstractKerasNetworkVisualizer class
    and overrides visualize_keras_model function to create a
    blueprint visualization with opaque nested models
    and the nested modules as a catalog.
    This class inherits from the SeeNN base class and overrides find_all_models
    function and visualize keras model function
    """

    def __init__(self, master_config, data_dict, base_path,
                    visualizer_config=None, logger=None):
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
        :param visualizer_config: The user-specified configuration dictionary
                for the visualizer. Default is None, meaning use all defaults.
        :param logger: A logger to send messaging to
        """

        SeeNN.__init__(self, master_config,
                        data_dict, base_path, suffix="_see_nn_modules",
                        visualizer_config=visualizer_config,
                        logger=logger)

        # Format for the Output
        self.format = self.visualizer_config['format']

        # initialize default colors for different
        # modules
        self.default_color_package = self.visualizer_config['default_color_package']

        # Check if multi (original Pineapple flow) colors need to be used
        self.use_multiple_colors_layers = self.visualizer_config['use_multiple_colors_layers']
        self.multiple_colors_layer_package = self.visualizer_config['multiple_colors_layer_package']

        # Initialize Layer shades for different layers
        self.layer_color_dict = self.visualizer_config['layer_color_dict']

        # Construct the Blueprint Graph
        self.module_catalog = pydot.Dot(graph_type='digraph',
                          label=self.visualizer_config['graph_label'],
                          fontsize=self.visualizer_config['fontsize'],
                          fontname=self.visualizer_config['fontname'],
                          rankdir=self.visualizer_config['rankdir'],
                          ranksep=self.visualizer_config['node_seperation_distance'],
                          rotate=self.visualizer_config['rotate'])

        # Dictionary to stor all the models
        self.models = {}

        # A list to store the edges in the graph
        # This is mainly to clean up repeated edges
        # that appear in the Keras JSON
        self.edges = []

        # Initialize Classes
        self.viz_utils = VisualizerUtils()
        self.graph_utils = GraphUtils()
        self.viz_shared_funcs = VisualizerSharedFunctions()


    def visualize_keras_model(self, keras_model, candidate):
        """
        Parses throught the candidate to construct the visualization of the
        network with nested modules attached and outputs the network as
        a png/pdf file.
        :param keras_model: The Keras model to visualize
        :param candidate: Dictionary representing the candidate to visualize
        :return: Nothing
        """
        # Get Visualizer Configs
        collapse_inputs = self.visualizer_config['collapse_inputs']
        write_class_names = self.visualizer_config['class_names']
        dimensions = [self.visualizer_config['layer_height'],
                     self.visualizer_config['layer_width']]
        show_constant_input = self.visualizer_config['show_constant_input']

        # Find all models
        # Get top-model
        top_model = candidate.get("interpretation").get("model")
        # Find all models

        self.models['top-model'] = [json.loads(top_model),
                                    self.viz_utils.generate_color(self.layer_color_dict,
                                    self.default_color_package,
                                    self.multiple_colors_layer_package,
                                    0, self.use_multiple_colors_layers)] # The zero here is
                                    # to fetch the first color from the color package

        self.models = self.viz_utils.find_all_models(self.models, self.layer_color_dict,
                                          self.default_color_package,
                                          self.multiple_colors_layer_package,
                                          self.use_multiple_colors_layers)

        # Refactor names for all the nested models
        for model in self.models:
            if model != "top-model":
                self.models[model][0] = self.viz_utils.refactor_layer_names_in_nested_model(
                                                                    self.models[model][0])

        # Check if activation functions need to be shown
        show_activation = self.visualizer_config['show_activation']

        # Conenct Softmod Block
        one_softmod_visualized = False
        for model in self.models:
            if model.find("softmod") == -1:
                continue
            if not one_softmod_visualized:
                self.module_catalog = self.viz_utils.connect_nested_module_internal_layers(model,
                                                           self.models,
                                                           self.module_catalog,
                                                           collapse_inputs,
                                                           write_class_names,
                                                           dimensions,
                                                           show_activation,
                                                           show_constant_input)
                one_softmod_visualized = True
            else:
                break

        # Connect all the nested module internal layers
        for model in self.models:
            # If model is not the blueprint module,
            # and at most one softmod has been added
            if model == 'top-model' \
            or model.find("softmod") != -1:
                continue
            self.module_catalog = self.viz_utils.connect_nested_module_internal_layers(model,
                                                       self.models,
                                                       self.module_catalog,
                                                       collapse_inputs,
                                                       write_class_names,
                                                       dimensions,
                                                       show_activation,
                                                       show_constant_input)


        # Blueprint connecttion color
        module_connection_color = self.visualizer_config['module_connection_color']
        # Add the Blueprint
        self.module_catalog = self.viz_shared_funcs.blueprint_layer_connectivity(self.models,
                                                                        self.module_catalog,
                                                                        show_constant_input,
                                                                        module_connection_color)

        # Write the grpah in the specified format
        self.write_pydot(self.module_catalog)



    def get_default_visualizer_config(self):
        """
        A chance for subclasses to supply a default configuration
        on top of which any user mods are made.
        :return: a dictionary populated with the default configuration
                for the visualizer.
        """

        default_config = {

            "format": "png",

            "fontsize": 16,
            "fontname": 'Roboto',
            "rankdir": "TB",

            # Specific offsets can be specified here
            # for different shades. Change the values
            # below 0 and 1. For best results we recommend
            # to keep the range between 0.1 - 0.3
            "layer_color_dict": {
                "InputLayer": 0.1,
                "Reshape": 0.12,
                "Conv1D": 0.13,
                "Conv2D": 0.17,
                "MaxPooling1D": 0.19,
                "MaxPooling2D": 0.20,
                "ZeroPadding3D": 0.22,
                "Flatten": 0.25,
                "AveragePooling2D": 0.27,
                "Dropout": 0.29,
                "Dense": 0.3,
                "Concatenate": 0.32,
                "Model": 0.34,
                "RepeatVector": 0.36,
                "Multiply": 0.38,
                "Add": 0.39,
                "Lambda": 0.4,
                "SpatialDropout1D": 0.41,
                "SpatialDropout2D": 0.44
            },
            # Please provide as many colors
            # as many models you expect.
            # This package will
            # generate random colors incase
            # colors fall short, but there
            # is no guarantee that they will be
            # pretty
            'default_color_package': [
                [0.586, 1.000, 1.000],
                [0.513, 0.141, 0.725],
                [0.094, 1.000, 1.000],
                [0.375, 0.739, 0.780],
                [0.967, 0.816, 0.961],
                [0.286, 1.000, 1.000],
                [0.750, 0.416, 0.961],
                [0.778, 0.631, 0.871],
                [0.613, 0.141, 0.725],
                [0.850, 0.539, 0.780],
                [0.186, 1.000, 1.000]
            ],
            "class_names": True,
            "graph_label": "Candidate Module Catalog",
            "node_seperation_distance": 0.4,

            'module_connection_color': 'black',
            'collapse_inputs': False,
            'layer_height': 0.5,
            'layer_width': 2,
            # 'condense_dropout_layer': False,

            # Specify if to use multiple color layers,
            # rather than shade
            'use_multiple_colors_layers': False,

            # If use_multiple_colors_layers is Fa,se,
            # provide the colors
            'multiple_colors_layer_package': {
                "InputLayer": "grey",
                "Reshape": "#F5A286",
                "Conv1D": "#F7D7A8",
                "Conv2D": "#F7D7A8",
                "MaxPooling1D": "#AADFA2",
                "MaxPooling2D": "#AADFA2",
                "ZeroPadding3D": "grey",
                "Flatten": "grey",
                "AveragePooling2D": "#A8CFE7",
                "Dropout": "#9896C8",
                "Dense": "#C66AA7",
                "Concatenate": "#F5A286",
                "Model": "#292D30",
                "RepeatVector": "grey",
                "Multiply": "grey",
                "Add": "grey",
                "Lambda": "#CAAFE7",
                "SpatialDropout1D": "#FFAAEE",
                "SpatialDropout2D": "#CAAFE7"
            },

            'show_activation': False,
            'rotate': 90,
            'show_constant_input': False

        }

        return default_config
