import numpy as np


class BaseGraphIntersection:
    """
    This class defines the data structures encoding the intersection of a pair of graph (g1, g2). 

    Formally, we consider graphs such that each vertex v come with some area/volume a_v attached to it.
    Then the intersection of g1 to g2 is way to know, for each vertex v in g_j, which are the vertices
    [w_1, w_2, ..., w_n] in g_i such that the a_v and a_w_k intersect for all k, and to know the proportion
    of a_v in this intersection.

    Given that the number n_v of vertices in g2 intersecting a vertex v of g1 may vary widely accross 
    vertices, we do not store those intersections using a 2D arrays. Instead, this information is stored
    using three one dimensional arrays, indexes_gi_to_gj, connections_gi_to_gj and weights_gi_to_gj 
    constructed as follows:
        the intersections of the vertex v of index k in gi can be gathered by first getting the 
        integers 

            p = indexes_gi_to_gj[k] and q = indexes_gi_to_gj[k + 1]. 

        Then, the integers

            [connections_gi_to_gj[p], 
             connections_gi_to_gj[p + 1], 
             ..., 
             connections_gi_to_gj[q - 1]]

        are the indexes in gj of the vertices that intersect v. The corresponding elements in
        weights_gi_to_gj give the 'proportion of intersection'.

    Define the basic attributes.
        graph_1: graph object
        graph_2: graph object

        indexes_g1_to_g2: 1D array of integers
        connections_g1_to_g2: 1D array of integers
        weights_g1_to_g2: 1D array of floats

        indexes_g2_to_g1: 1D array of integers
        connections_g2_to_g1: 1D array of integers
        weights_g2_to_g1: 1D array of floats

    """
    def __init__(self, graph_1=None, graph_2=None, **kwargs):
        if graph_1 is None:
            raise ValueError("No value has been given for kwargs 'graph_1'.")
        if graph_2 is None:
            raise ValueError("No value has been given for kwargs 'graph_2'.")
        self.graph_1 = graph_1
        self.graph_2 = graph_2

        self.indexes_g1_to_g2 = None  # this attributes work with the two below, see class description
        self.connections_g1_to_g2 = None
        self.weights_g1_to_g2 = None

        self.indexes_g2_to_g1 = None
        self.connections_g2_to_g1 = None
        self.weights_g2_to_g1 = None



