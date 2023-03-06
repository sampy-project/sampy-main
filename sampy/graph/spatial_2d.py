import numpy as np
from .spatial_functions import create_2d_coords_from_oriented_connection_matrix


class SpatialComponentsTwoDimensionalOrientedHexagons:
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_set_coords_from_vector(self, coord_first_vertex, vector, index_first_vertex=0,
                                            attribute_coord_x='coord_x', attribute_coord_y='coord_y'):
        if not hasattr(self, "connections"):
            raise ValueError("The graph object has no connection attribute.")
        if self.connections.shape[1] != 6:
            raise ValueError("The connections attribute of the graph object is not of shape (N, 6).")
        if index_first_vertex < 0 or index_first_vertex >= self.connections.shape[0]:
            raise ValueError("The given index_first_vertex is not the index of a vertex of the graph.")
        if not hasattr(self, 'df_attributes'):
            raise ValueError("The graph object has no attribute df_attributes.")

    # todo: make a version for disconnected graphs
    def set_coords_from_vector(self, coord_first_vertex, vector, index_first_vertex=0,
                               attribute_coord_x='coord_x', attribute_coord_y='coord_y'):
        """
        WARNING: The graph is assumed to be connected.

        Set the coordinates of the centroids of the hexagonal cells. The algorithm starts at the cell given by the
        kwarg 'index_first_vertex', which receives as coordinates the one given in 'coord_first_vertex'. Then we loop
        through each neighbours of the starting cell, giving coordinates to each one using the parameter 'vector' or a
        rotation of it (see description of 'vector' parameter below for a more detailed explanation). We then repeat the
        process with a vertex that has coordinates, and so on until each vertex has coordinates. This algorithm works
        only if the graph is connected.

        :param coord_first_vertex: couple of float. (x, y) coordinates of the first vertex.
        :param vector: couple of float (u, v) used to recursively construct de coordinates of each vertex. To understand
                       how, let us consider the vertex of index 'i' and assume it has coordinates (x, y). Then the
                       vertex connections[i, 0] will have as coordinates (x, y) + (u, v), the vertex connections[i, 1]
                       will have as coordinates (x, y) + rot(-pi/3)(u, v) (we rotate (u, v) by an anlge of pi/3
                       clockwise), the vertex connections[i, 2] will have as coordinates (x, y) + rot(-2 * pi/3)(u, v),
                       etc...
        :param index_first_vertex: optional, non-negative integer, default 0.
        :param attribute_coord_x: optional, string, default 'coord_x'. Name of the column of df_attributes in which to
                                  store the x coordinates.
        :param attribute_coord_y: optional, string, default 'coord_y'. Name of the column of df_attributes in which to
                                  store the y coordinates.
        """

        # we first create the six main directions where a neighbouring hexagon could be.
        list_vects = [np.array(vector)]
        for i in range(5):
            # the rotation is of -pi/3, cause the orientation is made clockwise.
            # the signs and such have been changes below using cos and sin parity.
            x = np.cos(np.pi / 3) * list_vects[i][0] + np.sin(np.pi / 3) * list_vects[i][1]
            y = - np.sin(np.pi / 3) * list_vects[i][0] + np.cos(np.pi / 3) * list_vects[i][1]
            list_vects.append(np.array([x, y]))

        coord_x, coord_y = create_2d_coords_from_oriented_connection_matrix(self.connections, index_first_vertex,
                                                                            coord_first_vertex, list_vects)

        self.df_attributes[attribute_coord_x] = coord_x
        self.df_attributes[attribute_coord_y] = coord_y
