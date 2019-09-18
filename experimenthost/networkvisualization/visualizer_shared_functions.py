from experimenthost.networkvisualization.visualizer_utils \
    import VisualizerUtils

class VisualizerSharedFunctions():
    """
    This Class contains functions that are shared by multiple visualizers
    """
    def __init__(self):
        self.viz_utils = VisualizerUtils()

    def blueprint_layer_connectivity(self, models_dict, graph, show_constant_input,
                                    module_connection_color):
        """
        This function Adds all the inputs and outputs
        in the top-model and then the nested module as
        opaque blocks and connects them

        :param models_dict: Dictionary containing the JSON of all the nested
        modules and the top-module
        :param graph: the blueprint pydot graph object

        """
        top_model = models_dict['top-model']
        top_model_base_color = models_dict['top-model'][1][0]
        top_model_layer_colors = models_dict['top-model'][1][1]
        top_model_layers = top_model[0]["config"]["layers"]

        nodes_in_graph = self.viz_utils.get_nodes_in_graph(graph)

        # Add all Nodes
        for layer in top_model_layers:
            layer_name = layer["name"]
            layer_class = layer["class_name"]
            if layer_name not in nodes_in_graph:
                if layer_class != "Model":
                    # If node is constant input, name the
                    # class Constant Input
                    if layer_name.find("constant") != -1:
                        layer_class = "Constant Input"
                    # If layers is softmod, color is black
                    # else fetch color from generated colors
                    if layer_name.find("softmod") != -1 \
                    or layer_name.find("constant") != -1:
                        color = module_connection_color
                    else:
                        color = top_model_layer_colors.get(layer_class,
                                                           top_model_base_color)
                    self.viz_utils.add_node(layer_name, graph, layer_class,
                                  color, [0.5, 2], show_constant_input)
                else:
                    model = models_dict[layer_name]
                    model_base_color = model[1][0]
                    # If it is a softmod Model
                    if layer_name.find("softmod") != -1:
                        self.viz_utils.add_node(layer_name, graph, "WeightedSum",
                                      module_connection_color, [2, 2], show_constant_input)
                    else:
                        self.viz_utils.add_node(layer_name, graph, layer_name,
                                      model_base_color, [2, 2], show_constant_input)

        # Add edges
        for layer in top_model_layers:
            layer_name = layer["name"]
            layer_class = layer["class_name"]
            inbound_nodes = self.viz_utils.get_inbound_nodes(layer)
            for inbound_node in inbound_nodes:
                self.viz_utils.add_edge(inbound_node, layer_name, module_connection_color, graph,
                                        show_constant_input=show_constant_input)



        return graph
