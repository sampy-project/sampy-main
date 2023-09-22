import numpy as np
from .jit_compiled_functions import convert_1D_float_like_array


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

            p = indexes_gi_to_gj[k]
            q = indexes_gi_to_gj[k + 1]

        Then, the integers

            [connections_gi_to_gj[p], 
             connections_gi_to_gj[p + 1], 
             ..., 
             connections_gi_to_gj[q - 1]]

        are the indexes in gj of the vertices that intersect v. The corresponding elements in
        weights_gi_to_gj give the 'proportion of intersection' between v and the associated vertex.

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

    def convert_1D_array(self, source_to_target, array_to_convert):
        """
        Converts a 1D array of float (or complex) arr made for one of the graph into an 
        array made for the other graph. For instance, if arr is made for g1, then:
            - arr.shape = (nb_vertices_g1,)
            - arr[i] is some quantities, encoded by a float (or which can be converted
              into a float), on the vertex i of g1
        Then the method would return an array r_arr such that:
            - r_arr.shape = (nb_vertices_g2,)
            - r_arr[j] is some quantities on the vertex j of g2, obtained using the
              intersection of g1 and g2

        WARNING: integer array will be converted into float array before intersecting.
                 This is done using the default numpy.asfarray method. Do the conversion
                 yourself prior to using this method if you need a specific dtype

        :param source_to_target: string, either 'g1_to_g2' or 'g2_to_g1'. Tells in
                                 which direction the conversion should be done
        :param array_to_convert: 1D array of float or complex numbers. If 
                                 source_to_target=='g1_to_g2', then it should be of 
                                 shape (nb_vertices_g1,). Else, it should be of 
                                 shape (nb_vertices_g2,).
        """
        if str(array_to_convert.dtype).startswith('int') or str(array_to_convert.dtype).startswith('uint'):
            array_to_convert = np.asfarray(array_to_convert)

        # start the convertion
        if source_to_target == 'g1_to_g2':

            # first we check that the required attributes have been created
            if self.indexes_g1_to_g2 is None:
                raise AttributeError("Intersection operation not performed, attribute 'indexes_g1_to_g2' missing.")
            if self.connections_g1_to_g2 is None:
                raise AttributeError("Intersection operation not performed, attribute 'connections_g1_to_g2' missing.")
            if self.weights_g1_to_g2 is None:
                raise AttributeError("Intersection operation not performed, attribute 'weights_g1_to_g2' missing.")
            
            returned_array = np.zeros((self.graph_2.number_vertices,), dtype=float)
            convert_1D_float_like_array(array_to_convert, returned_array, self.indexes_g1_to_g2, 
                                        self.connections_g1_to_g2, self.weights_g1_to_g2)
            return returned_array

        elif source_to_target == 'g2_to_g1':

            # first we check that the required attributes have been created
            if self.indexes_g2_to_g1 is None:
                raise AttributeError("Intersection operation not performed, attribute 'indexes_g2_to_g1' missing.")
            if self.connections_g2_to_g1 is None:
                raise AttributeError("Intersection operation not performed, attribute 'connections_g2_to_g1' missing.")
            if self.weights_g2_to_g1 is None:
                raise AttributeError("Intersection operation not performed, attribute 'weights_g2_to_g1' missing.")
            
            returned_array = np.zeros((self.graph_1.number_vertices,), dtype=float)
            convert_1D_float_like_array(array_to_convert, returned_array, self.indexes_g2_to_g1, 
                                        self.connections_g2_to_g1, self.weights_g2_to_g1)
            return returned_array

        else:
            raise ValueError("The parameter 'source_to_target' should be either 'g1_to_g2' or 'g2_to_g1'.")



