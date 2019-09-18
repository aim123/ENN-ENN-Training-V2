import json
import pydot
# Import Base class
from experimenthost.networkvisualization.abstract_keras_network_visualizer \
    import AbstractKerasNetworkVisualizer

from experimenthost.networkvisualization.visualizer_utils \
    import VisualizerUtils

from experimenthost.networkvisualization.graph_utils \
    import GraphUtils

class SeeNN(AbstractKerasNetworkVisualizer):
    """
    This class inherits from AbstractKerasNetworkVisualizer class
    and overrides visualize_keras_model function to create visualization as
    network png/pdf.
    Super constructor of base class is called and then other functionality
    is added (mainly including a color dictionary of layers, the graph that
    is built and a dictionary to store models)
    Some of the functionality of this class is derived from VisualizerUtils
    class and GraphUtils class
    """

    def __init__(self, master_config, data_dict, base_path,
                suffix="_see_nn_nested", visualizer_config=None,
                logger=None):
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

        # Call constructor of super class
        super(SeeNN, self).__init__(master_config,
                        data_dict, base_path, suffix=suffix,
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

        # Get the label
        self.label = self.visualizer_config['graph_label']
        # Construct the Graph
        self.graph = pydot.Dot(graph_type='digraph',
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


    def add_bluprint_layers(self, models_dict, graph, class_names, dimensions,
                            show_activation, show_constant_input):
        """
        This function takes in the top-model, finds it layers
        and connects those layers. If there are nested modules it
        also connects them to the layers in the blueprint
        :param models_dict: dictionary containing all the models
        :param graph: the blueprint pydot graph object
        :param class_names: Specifies if to use layer class
        names instead of actual names specified in JSON
        :param dimensions: Specifies the dimensions of the layer to be rendered
        :param show_activation: Determine weather to write activations or not
        :returns graph: with all the layers in the blueprint layer added
        and connected (does not connect the modules and layers)
        """
        top_model = models_dict['top-model']
        # Get the top-model color list containing
        # the base color and the layer shades
        top_model_color = top_model[1]
        # get the model
        top_model = top_model[0]

        # Get the layers of the model
        layers = top_model["config"]["layers"]
        # Loop through the layers
        for layer in layers:
            # If the layer is not a model
            if layer["class_name"] != "Model":
                # Get the layer name
                layer_name = layer["name"]
                # If label only layer's class name
                if class_names:
                    # Get the layer's information
                    layer_info = self.viz_utils.query_layer(layer_name,
                                                        models_dict)
                    # Get the layer's class name
                    layer_class = layer_info['class_name']
                    # If the layer is a a constant input layer,
                    #  manually specify the class name
                    if layer_name.find('constant_input') != -1:
                        layer_class = 'Constant Input'
                    # Depending on the class name
                    # find the the layer shade
                    # If the layer is a constant_input layer
                    # the color is black
                    model_color = top_model_color[1].get(layer_class, "black")
                else:
                    # If don't use class names for layers
                    # then use the layer name from the JSON
                    layer_class = layer_name
                    model_color = top_model_color[0]


                # Add the node to the graph
                graph = self.viz_utils.add_nodes(layer_name, graph,
                                                layer_class, model_color,
                                                dimensions, show_constant_input)

                # Add Blueprint Inbound Edges
                graph = self.connect_blueprint_inbounds(models_dict,
                                                        layer, graph,
                                                        class_names, dimensions,
                                                        show_activation, show_constant_input)
            else:
                # Add Softmod
                graph = self.connect_softmod_in_blueprint(models_dict,
                                                layer, graph, class_names,
                                                dimensions, show_activation, show_constant_input)

        return graph

    def connect_blueprint_inbounds(self, models_dict, layer, graph, class_names,
                                    dimensions, show_activation, show_constant_input):
        """
        This module connects the inbound nodes to a given layer at the
        blueprint level

        :param models_dict: dictionary containing all the models
        :param layer: The JSON of the Layer whose inbound nodes have to be
        connected
        :param graph: the blueprint pydot graph object
        :param class_names: Specifies if to use layer class
        names instead of actual names specified in JSON
        :param dimensions: Specifies the dimensions of the layer to be rendered
        :param show_activation: Determine weather to write activations or not
        :returns graph: with all the layers in the blueprint connected to the
        softmod modules
        """
        top_model = models_dict['top-model']
        # Get the top-model color list containing
        # the base color and the layer shades
        top_model_color = top_model[1]
        # Get the top-model base color
        top_model_base_color = top_model_color[0]
        # get the model
        top_model = top_model[0]
        # Get the layer name
        layer_name = layer["name"]
        # Get the inbound nodes and
        inbound_nodes = self.viz_utils.get_inbound_nodes(layer)
        if len(inbound_nodes):
            for inbound_node in inbound_nodes:
                if inbound_node.find('model') == -1:
                    layer_info = self.viz_utils.query_layer(inbound_node,
                                                        models_dict)['layer']

                    if show_activation:
                        # Get Layer Activation
                        if "activation" in list(layer_info["config"]):
                            layer_activation = layer_info["config"]["activation"]
                        else:
                            layer_activation = "linear"
                    else:
                        layer_activation = ""

                    if class_names:
                        layer_class = layer_info['class_name']
                        if inbound_node.find('constant_input') != -1:
                            layer_class = 'Constant Input'
                        model_color = top_model_color[1].get(layer_class, "black")
                    else:
                        layer_class = layer_name
                        model_color = top_model_color[0]
                    graph = self.viz_utils.add_nodes(inbound_node, graph,
                                                    layer_class,
                                                    model_color,
                                                    dimensions, show_constant_input)
                    # Add the edge between the inbound node and the node
                    graph = self.viz_utils.add_edge(inbound_node, layer_name,
                                                    top_model_base_color,
                                                    graph, layer_activation, show_constant_input)

        return graph

    def connect_softmod_in_blueprint(self, models_dict, layer, graph,
                                    class_names, dimensions, show_activation,
                                    show_constant_input):

        """
        This module connects all the softmod modules as single
        layers at the blueprint level

        :param models_dict: dictionary containing all the models
        :param layer: The JSON of the Softmod module
        :param graph: the blueprint pydot graph object
        :param class_names: Specifies if to use layer class
        names instead of actual names specified in JSON
        :param dimensions: Specifies the dimensions of the layer to be rendered
        :param show_activation: Determine weather to write activations or not
        :returns graph: with all the layers in the blueprint connected to the
        softmod modules
        """
        # If the layer is a softmod model
        model_name = layer["name"]
        if model_name.find("softmod") != -1:
            # Assign Softmod Color as black
            softmod_color = "black"
            # Add the softmod model to the graph
            self.viz_utils.add_nodes(model_name, graph, "WeightedSum",
                           softmod_color, dimensions, show_constant_input)
            # Get the inbound nodes to the softmod
            inbound_nodes = self.viz_utils.get_inbound_nodes(layer)
            if len(inbound_nodes):
                graph = self.connect_softmod_inbound_nodes(inbound_nodes,
                                                        models_dict, graph,
                                                        dimensions, class_names,
                                                        model_name, show_activation,
                                                        show_constant_input)

        return graph

    def connect_softmod_inbound_nodes(self, inbound_nodes, models_dict,
                                     graph, dimensions, class_names,
                                     model_name, show_activation, show_constant_input):
        """
        This module connects all the inbound nodes of softmod modules

        :param models_dict: dictionary containing all the models
        :param layer: The JSON of the Softmod module
        :param graph: the blueprint pydot graph object
        :param class_names: Specifies if to use layer class
        names instead of actual names specified in JSON
        :param dimensions: Specifies the dimensions of the layer to be rendered
        :param show_activation: Determine weather to write activations or not
        :returns graph: with all the layers in the blueprint connected to the
        softmod modules
        """
        top_model = models_dict['top-model']
        # Get the top-model color list containing
        # the base color and the layer shades
        top_model_color = top_model[1]
        # Get the top-model base color
        top_model_base_color = top_model_color[0]
        # get the model
        top_model = top_model[0]

        for inbound_node in inbound_nodes:
            # If the inbound node is not a model
            if inbound_node.find('model') == -1:
                # Get the layer's activation
                layer_info = self.viz_utils.query_layer(inbound_node,
                                            models_dict)['layer']

                if show_activation:
                    # Get Layer Activation
                    if "activation" in list(layer_info["config"]):
                        layer_activation = layer_info["config"]["activation"]
                    else:
                        layer_activation = "linear"
                else:
                    layer_activation = ""

                if class_names:
                    layer_class = layer_info['class_name']
                    if inbound_node.find('constant_input') != -1:
                        layer_class = 'Constant Input'
                    model_color = top_model_color[1].get(layer_class,
                                                         "black")
                else:
                    layer_class = layer_info['name']
                    model_color = top_model_color[0]

                graph = self.viz_utils.add_nodes(inbound_node, graph,
                                layer_class,
                               model_color,
                               dimensions, show_constant_input)
                # Add the edge between the inbound node and the node
                graph = self.viz_utils.add_edge(inbound_node, model_name,
                              top_model_base_color, graph,
                              layer_activation, show_constant_input)
        return graph


    def connect_nested_modules(self, models_dict, collapse_layers, graph, module_connection_color,
                                show_constant_input):
        """
        This function takes in the a dictionary containing all the
        models. The dictionary generated by the find_all_models function
        and connects all the models.
        :param models_dict: dictionary containing all the models
        :param graph: the blueprint pydot graph object
        :param module_connection_color: Color of the edge connecting
        different models
        :param collapse_layers: Removes input layers of
        nested modules
        :returns graph: with all the added edges between modules
        """
        # Loop over the models dict
        for model_name in models_dict:
            # If the model is not the top-model (blueprint)
            # and not a softmod module. We are ignoring the softmod modules
            # here as we don't want them connected as of now.
            # This function basically connects the first layer of a given
            # module to the last layer of the inbound module.
            #Since softmod is now collapsed, it is just added as single
            # layer and has no context for first and last layers.
            if model_name != 'top-model' \
            and model_name.find("softmod") == -1:
                # Get the models config
                model = models_dict[model_name][0]
                # Get all the inbound nodes
                inbound_nodes_to_model = self.viz_utils.get_inbound_nodes(model)
                # Loop over the inbound nodes
                for inbound_node in inbound_nodes_to_model:
                    # If the inbound node is not a model
                    if inbound_node.find('model') == -1 \
                    or inbound_node.find('softmod') != -1:
                        graph = self.connect_first_layer_model_to_inbound_node(models_dict,
                                                                            model,
                                                                            inbound_node,
                                                                            module_connection_color,
                                                                            graph,
                                                                            collapse_layers,
                                                                            show_constant_input)
                    else:
                        graph = self.connect_first_layer_model_to_inbound_last_layer(models_dict,
                                                                            model,
                                                                            inbound_node,
                                                                            module_connection_color,
                                                                            graph,
                                                                            collapse_layers,
                                                                            show_constant_input)

        return graph

    def connect_first_layer_model_to_inbound_node(self, models_dict,
                                            model, inbound_node,
                                            module_connection_color, graph,
                                            collapse_layers, show_constant_input):
        """
        This function Connects the First layer of the model to the inbound layer
        :param models_dict: dictionary containing all the models
        :param model: Model whose first layer is to be connected
        :param inbound_node: Layer to which the model is to be connected
        :param graph: the blueprint pydot graph object
        :param module_connection_color: Color of the edge connecting
        different models
        :param collapse_layers: Removes input layers of
        nested modules
        :returns graph: with all the added edges between modules
        """
        # Connect the First layer of the model to the inbound node
        first_layer_of_model = self.viz_utils.get_layer_name(0, model)
        if not collapse_layers:
            graph = self.viz_utils.add_edge(inbound_node, first_layer_of_model,
                          module_connection_color, graph,
                          show_constant_input=show_constant_input)

        else:
            outbound_nodes = self.viz_utils.find_nodes_to_inbound_layer(
                                                    first_layer_of_model,
                                                    models_dict)
            for outbound_node in outbound_nodes:
                graph = self.viz_utils.add_edge(inbound_node, outbound_node,
                              module_connection_color, graph,
                              show_constant_input=show_constant_input)

        return graph

    def connect_first_layer_model_to_inbound_last_layer(self,
                                                    models_dict,
                                                    model, inbound_node,
                                                    module_connection_color,
                                                    graph,
                                                    collapse_layers,
                                                    show_constant_input):
        """
        Connect the First layer of the model to the last layer of
        inbound model
        :param models_dict: dictionary containing all the models
        :param model: Model whose first layer is to be connected
        :param inbound_node: Inbound model whose last layer is to be connected
        :param graph: the blueprint pydot graph object
        :param module_connection_color: Color of the edge connecting
        different models
        :param collapse_layers: Removes input layers of
        nested modules
        :returns graph: with all the added edges between modules
        """
        first_layer_of_model = self.viz_utils.get_layer_name(0, model)
        if not collapse_layers:
            last_layer_of_inbound_model = self.viz_utils.get_layer_name(-1,
                                            models_dict[inbound_node][0])
            graph = self.viz_utils.add_edge(last_layer_of_inbound_model,
                         first_layer_of_model,
                         module_connection_color, graph,
                         show_constant_input=show_constant_input)

        else:
            outbound_nodes = self.viz_utils.find_nodes_to_inbound_layer(
                                                    first_layer_of_model,
                                                    models_dict)
            for outbound_node in outbound_nodes:
                last_layer_of_inbound_model = self.viz_utils.get_layer_name(-1,
                                                models_dict[inbound_node][0])
                graph = self.viz_utils.add_edge(last_layer_of_inbound_model,
                              outbound_node,
                              module_connection_color, graph,
                              show_constant_input=show_constant_input)

        return graph


    def compile_blueprint(self, models_dict, collapse_layers, graph,
                            module_connection_color, show_constant_input):
        """
        This function calls connect_nested_modules to conenct the models
        in the graph. It also connects the layers of the top-module to the
        nested modules
        :param models_dict: dictionary containing all the models
        :param graph: the blueprint pydot graph object
        :param module_connection_color: Color of the edge connecting
        different models
        :param collapse_layers: Removes input layers of
        nested modules
        :returns graph: with all layers and modules in blueprint
        connected and added
        """
        # Connect All Nested Modules
        graph = self.connect_nested_modules(models_dict, collapse_layers,
                                            graph, module_connection_color,
                                            show_constant_input)

        # Connect layers of top-module to nested modules
        top_model = models_dict["top-model"][0]
        # Get the layers of the top-model
        top_model_layers = top_model["config"]["layers"]
        # Loop over the layers
        for layer in top_model_layers:
            # If the layer is not a model
            if layer["class_name"] != "Model":
                # Get the inbound layers to the layer
                layer_inbound_nodes = self.viz_utils.get_inbound_nodes(layer)
                # If there are any inbound layers
                if len(layer_inbound_nodes):
                    # loop over the inbound layers
                    for inbound_node in layer_inbound_nodes:
                        # If the inbound layer is not a model
                        if inbound_node.find("model") != -1 \
                        and inbound_node.find("softmod") == -1:
                            # Find the last layer of the current model
                            last_layer_of_model = models_dict[inbound_node][0] \
                                                            ["config"]["layers"] \
                                                            [-1]['name']

                            # Add an edge
                            graph = self.viz_utils.add_edge(last_layer_of_model,
                                                         layer["name"],
                                                         module_connection_color,
                                                         graph,
                                                         show_constant_input=show_constant_input)

                        else:
                            graph = self.viz_utils.add_edge(inbound_node,
                                                        layer["name"],
                                                        module_connection_color,
                                                        graph,
                                                        show_constant_input=show_constant_input)

            elif layer["name"].find("softmod") != -1:
                layer_inbound_nodes = self.viz_utils.get_inbound_nodes(layer)
                # If there are any inbound layers
                if len(layer_inbound_nodes):
                    # loop over the inbound layers
                    for inbound_node in layer_inbound_nodes:
                        # If the inbound layer is not a model
                        # and not a softmod
                        if inbound_node.find("model") != -1 \
                        and inbound_node.find("softmod") == -1:
                            # Find the last layer of the current model
                            last_layer_of_model = models_dict[inbound_node][0] \
                                                            ["config"] \
                                                            ["layers"][-1][ \
                                                            'name']
                            # Add an edge
                            graph = self.viz_utils.add_edge(last_layer_of_model,
                                                            layer["name"],
                                                            module_connection_color,
                                                            graph,
                                                            show_constant_input=show_constant_input)

                        else:
                            graph = self.viz_utils.add_edge(inbound_node,
                                                            layer["name"],
                                                            module_connection_color,
                                                            graph,
                                                            show_constant_input=show_constant_input)

        return graph

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
        module_connection_color = self.visualizer_config['module_connection_color']
        write_class_names = self.visualizer_config['class_names']
        dimensions = [self.visualizer_config['layer_height'],
                     self.visualizer_config['layer_width']]
        show_constant_input = self.visualizer_config['show_constant_input']

        # Find all models
        # Get top-model
        top_model = candidate.get("interpretation").get("model")
        # Find all models

        # Find the top model (the blueprint) and assign it
        # its base color and color shades.
        self.models['top-model'] = [json.loads(top_model),
                                    self.viz_utils.generate_color(self.layer_color_dict,
                                    self.default_color_package,
                                    self.multiple_colors_layer_package,
                                    0, self.use_multiple_colors_layers)] # The zero
                                    # here is to fetch the first color from the
                                    # color package
        # Here we call the function find all models
        # to assign colors to all the other models
        # nested within the blueprint model
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
        # Connect all the nested module internal layers
        for model in self.models:
            if model != 'top-model' \
            and model.find("softmod") == -1:
                self.graph = self.viz_utils.connect_nested_module_internal_layers(model,
                                                           self.models,
                                                           self.graph,
                                                           collapse_inputs,
                                                           write_class_names,
                                                           dimensions,
                                                           show_activation,
                                                           show_constant_input)


        # Add bluepring level (top-model) layers
        self.graph = self.add_bluprint_layers(self.models, self.graph,
                                write_class_names, dimensions, show_activation,
                                show_constant_input)
        # Connect the models to thr blueprint layer
        # with the speicifed connection color
        self.graph = self.compile_blueprint(self.models, collapse_inputs,
                              self.graph, module_connection_color, show_constant_input)

        # Write the grpah in the specified format
        self.write_pydot(self.graph)

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
            "graph_label": "Nested SeeNN",
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
