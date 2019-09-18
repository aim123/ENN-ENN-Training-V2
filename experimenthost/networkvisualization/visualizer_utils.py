import pydot
class VisualizerUtils():
    """
    This class contain the functions that the visualizer utilizes
    for color generation and extracting useful information about
    the layer from the JSON. It also contains other helper functions
    like refactoring JSON, pydot utils, adding the legend, etc.
    """
    NO_LABEL = ""
    def __init__(self):
        self.edges = []

    def find_all_models(self, models_dict,
                        layer_color_dict,
                        default_color_package,
                        multiple_color_layer_package,
                        use_multiple_color_layer_package):
        """
        Finds all the sub-models nested within the candidate and
        adds them to the dictionary of models. This function
        loops over the candidate dictionary and appends each nested model's
        JSON to a dictionary.
        :param models_dict: Dictionary containing the JSON of all the nested
        modules and the top-module
        :param layer_color_dict: Dictionary containing shades (offset
        of saturation by values 0-1) for different layers
        :param default_color_package: List of colors provided
        for different modules
        :param multiple_color_layer_package: Dictionary containing
        layer colors (Pineapple Flow Original Colors)
        :param use_multiple_color_layer_package: Specifies if to use the
        multi colors for different layers. If False, layer
        shades will be used
        :returns models_dict: Dictionary containing the JSON of nested
        modules with the base colors and the shades
        """
        # Index to fetch colors from default_color_package
        # Start at 1
        color_index = 1
        # Get the model layers
        model_layers = models_dict['top-model'][0]["config"]["layers"]
        # Loop over the layers
        for layer in model_layers:
            # If the layer is a model
            if layer["class_name"] == "Model":
                # Get the model's name
                model_name = layer["name"]
                # Generate a base color and
                # layer shades for the model
                if color_index > len(default_color_package) - 1:
                    color_index = 1
                color = self.generate_color(layer_color_dict,
                                            default_color_package,
                                            multiple_color_layer_package,
                                            color_index,
                                            use_multiple_color_layer_package)
                if model_name.find("softmod") == -1:
                    # Update the color Index
                    color_index += 1
                # Add the model to the dictionary
                # with the color
                models_dict[model_name] = [layer, color]

        return models_dict

    def query_layer(self, layer_name, models_dict):
        """
        This function returns a dictionary
        containing useful information
        about a layer
        :param layer_name: A string containing
        the name of the layer
        :returns : A dictionary
        containing the model to which the
        layer belongs, the layer's class
        and the layer's JSON.
        If layer is not found, an empty dictionary
        is returned
        """
        # Loop over the models
        for model in models_dict:
            # Get the model layers
            model_layers = models_dict[model][0]["config"]["layers"]
            # Loop over the layers
            for layer in model_layers:
                # If the layer queried is
                # the current layer
                if layer["name"] == layer_name:
                    # Return the dictionary
                    # containing the model
                    # and the layer JSON
                    layer_class = layer["class_name"]
                    return {"model": model,
                            "class_name": layer_class,
                            "layer": layer}
        # If layer not found return
        # an empty dict
        return {}

    # Visualizer Utils class
    def validate_color_format(self, hue, saturation, value):
        """
        This function validates the given hue, saturation and value strings
        according to the format that Pydot requires
        :param hue: string containing the hue of the color
        :param saturation: string containing the saturation of the color
        :param value: string containing the value of the color
        :returns correctly formated string for pydot
        """
        # Check if the types of the arguments are strings
        assert isinstance(hue, str) \
        and isinstance(saturation, str) \
        and isinstance(value, str)

        # Return the correct format - 0.hhh 0.sss 0.vvv
        hue = hue[:5].ljust(5, '0')
        saturation = saturation[:5].ljust(5, '0')
        value = value[:5].ljust(5, '0')
        return "{0} {1} {2}".format(hue, saturation, value)

    # Visualizer Utils class
    def generate_color(self,
                       layer_color_dict,
                       default_color_package,
                       multiple_color_package,
                       color_index,
                       use_multi_colors):
        """
        This method generates assigns the model a color
        from the color package specified. It also assigns
        the model different layers color based on an offset
        saturation from the base color
        :param layer_color_dict: Dictionary containing shades (offset
        of saturation by values 0-1) for different layers
        :param default_color_package: List of colors provided
        for different modules
        :param multiple_color_package: Dictionary containing
        layer colors (Pineapple Flow Original Colors)
        :param color_index: Keeps track of module's used to
        reference color
        :param use_multi_colors: Specifies if to use the
        multi colors for different layers. If False, layer
        shades will be used
        :returns : A list containing the module base color,
        and a dictionary of the different shades of the layers
        for that module
        """
        # Get a new base color from the color_package specified
        color = default_color_package[color_index]
        # Check if the color retrived is in the color format,
        # and if not return a correctly formatted string that
        # pydot can use
        model_color = self.validate_color_format(str(color[0]), str(color[1]), str(color[2]))

        # Specify an empty dictionary that can
        # store the different colors for the
        # layers based off of the base color
        model_color_dict = {}
        # Loop over the layer types
        for key in list(layer_color_dict.keys()):
            # Add the offset to the existing saturation from
            # the layer offset specifed in layer_color_dict
            color[1] = color[1] + layer_color_dict[key]

            # If the new saturation is larger than 1,
            # Divide it by 10
            if color[1] > 1:
                color[1] /= 10

            # Valudate the new layer color generated
            model_color_dict[key] = self.validate_color_format(str(color[0]),
                                                              str(color[1]),
                                                              str(color[2]))

        # Return a list containg the model base color
        # And the different layer colors based of the
        # base color or the multicolor layers
        if use_multi_colors:
            model_color_dict = multiple_color_package
        return [model_color, model_color_dict]



    # Visualizer Utils class
    def refactor_layer_names_in_nested_model(self, model_dict):
        """
        This function loops over the layers of a nested model
        and appends the model name to the layer_name
        :param models_dict: Dictionary containing the JSON of all the nested
        modules and the top-module
        :returns model_dict: Dictionary with the refactored names contained
        within the namespace of each nested module
        """
        # Get the model name
        model_name = model_dict["name"]
        # Get Models output layers
        model_output_layers = model_dict['config']['output_layers'][0][0]
        # Append Model Name to the output name
        model_output_layers = '{0}_{1}'.format(model_name, model_output_layers)
        # Assign it back to the original dict
        model_dict['config']['output_layers'][0][0] = model_output_layers


        # Get the model layers except the
        # input layer
        model_layers = model_dict["config"]["layers"]
        for layer in model_layers:
            layer["name"] = '{0}_{1}'.format(model_name, layer["name"])
            if len(layer['inbound_nodes']):
                inbound_nodes = layer['inbound_nodes'][0]
                if len(inbound_nodes):
                    for inbound_node in inbound_nodes:
                        inbound_node[0] = '{0}_{1}'.format(model_name,
                                                          inbound_node[0])

        return model_dict


    def find_layers_in_nested_module(self, model):
        """
        Given a model this function returns a dictionary
        containing the model number and a list of layers and their
        inbound nodes
        :param model: the model JSON
        :returns layer_dict: A dictionary consisting of the model name,
        its layers and each layer's inbound nodes
        """
        # Get the layers of the model
        layers = model["config"]["layers"]

        # Create an empty dictionary to store the layers
        layers_dict = {}
        # Get the model name and store it in the
        # layers_dict
        layers_dict['model'] = model["name"]
        # Assign an empty list to the 'layers' key
        # in the dictionary
        layers_dict["layers"] = []

        # Loop through the layers in the model
        for layer in layers:
            # Get the layer name
            layer_name = layer["name"]
            # To the list in the 'layers' key
            # append the layer name and its inbound nodes
            layers_dict["layers"].append({layer_name: self.get_inbound_nodes(layer)})

        # Return the dictionary
        return layers_dict


    def connect_nested_module_internal_layers(self, model_name, models_dict, graph,
                                              collapse_layers,
                                              class_names, dimensions,
                                              show_activation, show_constant_input):
        """
        This function takes in the model, finds it layers
        and connects the internal layers.
        :param models_dict: Dictionary containing the JSON of all the nested
        modules and the top-module
        :param model: the model dictionary that contains
        the model JSON and the color information. The dict
        created by find all models
        :param graph: the blueprint pydot graph object
        :param collapse_layers: Boolean to remove input layers of
        nested modules
        :param class_names: Specifies if to use layer class
        names instead of actual names specified in JSON
        :param dimensions: Specifies the dimensions of the layer to be rendered,
        :param show_activation: Specifies if activation of layers shoudl be shown
        :returns nothing
        """
        # Get the model
        model = models_dict[model_name]
        # Get the list containing the color
        # information for the model base color
        # and the shade for each layer
        model_color_global = model[1]
        # Find the model base color
        model_color_base_color = model_color_global[0]
        # Get the model
        model = model[0]

        # Find layers of the given model
        model_name = model["name"]
        layers_in_model = self.find_layers_in_nested_module(model)["layers"]
        for layer in layers_in_model:
            # Find the node layer
            node = list(layer.keys())
            # Get layer class
            layer_class = self.query_layer(node[0], models_dict)['class_name']

            # If layers is softmod, color is black
            # else fetch color from generated colors
            if model_name.find("softmod") != -1:
                model_color = "black"
            else:
                # Get model color layer shade
                model_color = model_color_global[1].get(layer_class, "blue")

            if not class_names:
                layer_class = []
            # Add the node
            if collapse_layers and node[0].find("input") != -1:
                continue

            graph = self.add_nodes(node, graph, layer_class, model_color, dimensions)

            # Find the inbound nodes
            inbound_nodes = list(layer.values())[0]

            # Delete inbound nodes that are inputs
            # if collapse_layers is turend on
            if collapse_layers:
                # Lambda function filters the elements
                # that do not have a sub-string of input
                # in the list inbound_nodes
                inbound_nodes = list(filter(lambda x: x.find("input") == -1, inbound_nodes))

            # find the class names of the inbound nodes
            # and store them in a list
            inbound_nodes_classes = []
            if class_names:
                # By looping over all inbound nodes
                for inbound_node in inbound_nodes:
                    # The class name of a inbound node
                    layer_info = self.query_layer(inbound_node, models_dict)
                    # Get Class name
                    layer_class_in_node = layer_info['class_name']
                    # Append in inbound_nodes_classes
                    inbound_nodes_classes.append(layer_class_in_node)


            # Add the inbound nodes
            graph = self.add_nodes(inbound_nodes, graph, inbound_nodes_classes,
                                   model_color,
                                   dimensions)

            graph = self.connect_nested_module_internal_layers_edges(models_dict, node, graph,
                                                           inbound_nodes, show_activation,
                                                           model_color_base_color,
                                                           show_constant_input)


        return graph

    def connect_nested_module_internal_layers_edges(self, models_dict, node, graph,
                                                    inbound_nodes, show_activation,
                                                    model_color_base_color,
                                                    show_constant_input):
        """
        This function serves as a helper to connect_nested_module_internal_layers.
        It is responsible to connect all the inbound edges of the layers of the internal
        modules.
        :param models_dict: Dictionary containing the JSON of all the nested
        modules and the top-module
        :param node: The destination name
        :param inbound_nodes: list of nodes that have to be connected to the node
        :param graph: the blueprint pydot graph object
        :param show_activation: Specifies if activation of layers shoudl be shown
        :returns nothing
        """
        # Add all the edges from the inbound nodes
        # to the present node
        for inbound_node in inbound_nodes:
            layer_info = self.query_layer(inbound_node, models_dict)['layer']
            # Check if activation is required
            if show_activation:
                # Get Layer Activation
                if "activation" in list(layer_info["config"]):
                    layer_activation = layer_info["config"]["activation"]
                else:
                    layer_activation = "linear"
            else:
                layer_activation = ""
            graph = self.add_edge(inbound_node, node[0], model_color_base_color, graph,
                        layer_activation, show_constant_input)

        return graph

    def get_nodes_in_graph(self, graph):
        """
        This function takes in a graph
        and returns a list of nodes in the graph
        :param graph: the blueprint pydot graph object
        :returns: A list containing the nodes in the graph
        """
        return [node.get_name() for node in graph.get_nodes()]

    def add_nodes(self, nodes, graph, labels, color, dimensions,
                show_constant_input=False):
        """
        Given a list of nodes, the function adds the nodes to the
        graph
        :param nodes: A list containing the nodes to be added
        :param graph: the blueprint pydot graph object
        :param labels: A list containing the label to be displayed for each of
        the node
        :param color: The color of the nodes
        :param dimensions: Specifies the dimensions of the layer to be rendered
        :returns graph: New graph with the nodes added
        """

        # Get the list of all the nodes already
        # in the graph
        nodes_in_graph = self.get_nodes_in_graph(graph)

        # Check if the nodes input is a list
        if isinstance(nodes, list):
            # Loop over all the nodes
            for i, _ in enumerate(nodes):

                # Get the current node
                node = nodes[i]
                # If labels are not empty
                if len(labels):
                    # Get the label
                    # for that node
                    label = labels
                    if label is list:
                        label = label[i]
                else:
                    # Else use the node name instead
                    # of the label
                    label = node

                # If node is a nested list
                # keep indexing into the 0th position
                while isinstance(node, list):
                    node = node[0]

                # If node already does not exist,
                # Add it to the Graph
                if node not in nodes_in_graph:
                    graph = self.add_node(node, graph, label, color, dimensions,
                    show_constant_input)
        else:
            # If node is not a list
            # just add that node
            # with the label

            # If node already does not exist,
            # Add it to the Graph
            if nodes not in nodes_in_graph:
                graph = self.add_node(nodes, graph, labels, color, dimensions,
                show_constant_input)

        return graph

    def add_node(self, layer, graph, label, color, dimensions,
                show_constant_input=False):
        """
        Adds the given node to the graph with appropriate color
        NOTE: This function does not validate for an existing layer
        in the graph. To validate please use the add_nodes function and
        pass the layer as the node param: [layer]
        :param layer: Layer to be added to the graph
        :param graph: the blueprint pydot graph object
        :param labels: A list containing the label to be displayed for each of
        the node
        :param color: The color of the node
        :param dimensions: Specifies the dimensions of the layer to be rendered
        :returns graph: with the node added
        """
        # If constant_input continue
        if not show_constant_input \
        and layer.find("constant") != -1:
            return graph
        # Decide fontcolor
        if layer.find("softmod") != -1 \
        or layer.find("constant") != -1:
            fontcolor = "white"
        else:
            fontcolor = "black"
        # Create a node with the layer as the identifier
        # and label as the text to be displayed for that
        # node
        node = pydot.Node(layer,
                  label=label,
                  xlabel="",
                  xlp=5,
                  orientation=0,
                  height=dimensions[0],
                  width=dimensions[1],
                  fontsize=10,
                  shape='box',
                  style='rounded, filled',
                  color=color,
                  fontcolor=fontcolor)
        # Add the node to the graph
        graph.add_node(node)

        return graph

    def add_edge(self, src, dst, color, graph, label=NO_LABEL, show_constant_input=False):
        """
        Add an edge from src -> dst in the graph
        :param src: A string consisting of the name of the source node
        :param dst: A string consisting of the name of the destination node
        :param graph: the graph to which the edge must be added
        :param color: The color of the edge
        :returns graph: with the edge added
        """
        # If constant_input continue
        if src.find("constant") != -1:
            if not show_constant_input:
                return graph
        # Create an edge object
        edge = pydot.Edge(src, dst, color=color, label=label,
                        fontcolor="#979ba1",
                         fontsize=10)
        # Check if the list containing
        # existing edges is empty
        if not len(self.edges):
            # If not then add the edge

            # Add the edge to the graph
            graph.add_edge(edge)
            # Append the string of the
            # format src -> dst to the
            # list of existing edges
            self.edges.append(src + '->' + dst)
        else:
            # If the list of existing
            # edges is not empty

            # Create a string for the edge
            # of format src -> dst
            edge_string = src + '->' + dst
            # If edge is not in existing edges
            if edge_string not in self.edges:
                # Add the edge to the graph
                graph.add_edge(edge)
                # Append the string of the
                # format src -> dst to the
                # list of existing edges
                self.edges.append(edge_string)

        return graph

    def get_inbound_nodes(self, layer):
        """
        This function finds all the inbound nodes belonging to a specific
        layer/model
        :param layer: The JSON description of the layer
        :returns in_layers_clean: A list containing all the inbound nodes to
        that layer
        NOTE: The layer can actually be a modelConfig as models also have
        inbound_nodes
        """
        # Get the class name of the layer/model
        layer_name = layer["name"]
        # Get the class name of the layer/model
        layer_class_name = layer["class_name"]
        # Get all the inbound nodes of the layer/model
        inbound_layers = layer["inbound_nodes"]
        # If inbound nodes exist
        if len(inbound_layers):
            # And if it is a layer and not a model
            if layer_class_name != "Model":
                # Index into the 0th index (due to structure of JSON)
                inbound_layers = inbound_layers[0]
            else:
                # If the layer is not a softmod layer
                if layer_name.find('model_softmod') != -1:
                    inbound_layers = inbound_layers[0]
        # Else just return an empty list
        else:
            return []

        # Create an empty list to store the inbound nodes
        in_layers_clean = []
        # Loop over the inbound_nodes found
        for inbound_layer in inbound_layers:
                # If inbound_layer is not empty
            if len(inbound_layer):
                # Keep indexing into the 0th index
                # until you have the layer name (type=str) ->
                # (due to structure of JSON)
                while isinstance(inbound_layer, list):
                    inbound_layer = inbound_layer[0]
                # Append that inobund layer to the list
                in_layers_clean.append(inbound_layer)
        # Return that list
        return in_layers_clean


    def find_nodes_to_inbound_layer(self, inbound_layer, models_dict):
        """
        Returns a list of all the layers outbound to a node
        :param models_dict: dictionary containing all the models
        :param inbound_layer: The layer name whose outbound nodes have to
        be queried
        :returns layers: list containing outbound nodes to a given node
        """
        # create a list of empty layers
        layers = []
        # loop over the models
        for model_name in models_dict:
            # Get the models JSON
            model = models_dict[model_name][0]
            # Get the models layers
            for layer in model["config"]["layers"]:
                # Loop over the models inbound layers
                if inbound_layer in self.get_inbound_nodes(layer):
                    # Append the layer's name to the layers list
                    layers.append(layer["name"])

        # Return the layers list
        return layers

    def get_input_layers(self, model):
        """
        This function finds the input layers
        to the model provided
        :param model: JSON of the model
        :returns a list of input_layers
        """
        input_layers = model["config"]['input_layers']
        return [input_layer[0] for input_layer in input_layers]

    def get_output_layers(self, model):
        """
        This function finds the output layers
        to the model provided
        :param model: JSON of the model
        :returns a list of ouput_layers
        """
        output_layers = model["config"]['output_layers']
        return [output_layer[0] for output_layer in output_layers]


    def get_layer_name(self, layer_index, model):
        """
        Given the model JSON and the layer index
        return the layer_name
        :param layer_index: Represents layer number of the model
        :param model: The model JSON
        :returns layer_name: The name of the layer
        """
        # Get layer config
        layer_config = model["config"]
        # Get the layers
        layers = layer_config["layers"]
        # Get the name of the layer requested
        layer_name = layers[layer_index]["name"]

        return layer_name
