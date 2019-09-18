from experimenthost.networkvisualization.visualizer_utils \
import VisualizerUtils

class GraphUtils():
    """
    This class contains functions that are required
    to traverse the pydot graph
    """
    def __init__(self):
        """
        Constructor for the class. This just initializes
        an object of vizualizer utils
        """
        self.viz_utils = VisualizerUtils()

    def return_nodes_src_to_dst(self, models_dict, graph,
                                src='constant_input',
                                dst='reshape_softmod'):

        """
        This function implements a depth first search
        for the pydot graph object

        :param graph: the blueprint pydot graph object
        :param src: Node in the graph from where the search
        must start
        :param dst: Node in the graph that is to be searched
        down from the src
        :returns nodes_on_the_way: A nested list containing
        all the nodes along different paths from src
        to dst
        """
        # Create a list to collect the nodes
        # on the path
        nodes_on_the_way = []
        # Get the JSON of the src
        layer = self.viz_utils.query_layer(src,
                                           models_dict)
        # Get the layer's name
        layer_name = layer['layer']['name']

        # If the current layer is not
        # the node we are looking for
        if layer_name.find(dst) == -1:
            # Find all the outbound edges
            # of the layer
            start_edges = self.find_edge_with_src(layer_name, graph)

            # Loop through all the outbound edges
            for edge in start_edges:
                # Retrieve the destination
                destination = edge.get_destination()
                # Recursively call this function
                # Witht the new destination
                nodes = self.return_nodes_src_to_dst(graph, destination, dst)

                # Append the nodes found
                nodes_on_the_way.append(nodes)
                nodes_on_the_way.append(str(destination))
            # Return the list of the nodes found
            return_val = nodes_on_the_way
        else:
            # If the current layer is the layer
            # we are looking for, return the name
            # of the layer
            return_val = str(layer_name)
        return return_val

    def find_edge_with_src(self, source, graph):
        """
        This function founds all the nodes outbound
        of from the source node

        :param source: name of the node whose outbound
        nodes have to be found
        :param graph: the pydot graph object in which
        to search

        :returns edges: A list containing the pydot
        edge objects of all the outbound nodes found
        """
        # Get all the edges in the graph
        all_edges = graph.get_edges()
        edges = []
        # Loop over all the edges
        for edge in all_edges:
            # If edge's source is the
            # required source
            if edge.get_source() == source:
                # Add that edges to the list
                edges.append(edge)
        # Return the edf
        return edges
