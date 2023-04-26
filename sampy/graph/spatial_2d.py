import numpy as np
from .spatial_functions import create_2d_coords_from_oriented_connection_matrix


class SpatialComponentsTwoDimensionalOrientedHexagons:
    """
    Allow user to introduce "spatial components" to graphs based on Oriented Hexagons. Those new components are not
    generated from the start, as most user may not need them, and users should
    """
    def __init__(self, **kwargs):
        self.cell_vertices = None
        self.index_cell_vertices = None

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

        The results are stored as attributes in the dataframe df_attributes of the graph.

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
        list_directions_to_neighbours = [np.array(vector)]
        for i in range(5):
            # the rotation is of -pi/3, cause the orientation is made clockwise.
            # the signs and such have been changes below using cos and sin parity.
            x = np.cos(np.pi / 3) * list_directions_to_neighbours[i][0] + \
                np.sin(np.pi / 3) * list_directions_to_neighbours[i][1]
            y = - np.sin(np.pi / 3) * list_directions_to_neighbours[i][0] + \
                np.cos(np.pi / 3) * list_directions_to_neighbours[i][1]
            list_directions_to_neighbours.append(np.array([x, y]))

        coord_x, coord_y = create_2d_coords_from_oriented_connection_matrix(self.connections,
                                                                            index_first_vertex,
                                                                            coord_first_vertex,
                                                                            list_directions_to_neighbours)

        self.df_attributes[attribute_coord_x] = coord_x
        self.df_attributes[attribute_coord_y] = coord_y

    def _sampy_debug_create_hexagon_vertices(self, attribute_coord_x='coord_x', attribute_coord_y='coord_y',
                                             list_directions_to_neighbours=None, list_directions_to_hex_vert=None):
        if not hasattr(self, 'df_attributes'):
            raise ValueError("The graph object has no attribute df_attributes.")
        if attribute_coord_x not in self.df_attributes.list_col_name:
            raise ValueError("The dataframe df_attributes in the graph has no column named " + attribute_coord_x + ", "
                             "which is supposed to be the name of the column containing x coord of hexagon centroids.")
        if attribute_coord_y not in self.df_attributes.list_col_name:
            raise ValueError("The dataframe df_attributes in the graph has no column named " + attribute_coord_y + ", "
                             "which is supposed to be the name of the column containing x coord of hexagon centroids.")
        if list_directions_to_neighbours is not None:
            nb_item = 0
            for _ in list_directions_to_neighbours:
                nb_item += 1
            if nb_item != 6:
                raise ValueError("The kwarg 'list_directions_to_neighbours' should be an iterable with 6 items ("
                                 "preferably a list).")
        if list_directions_to_hex_vert is not None:
            nb_item = 0
            for _ in list_directions_to_hex_vert:
                nb_item += 1
            if nb_item != 6:
                raise ValueError("The kwarg 'list_directions_to_hex_vert' should be an iterable with 6 items ("
                                 "preferably a list).")

    def create_hexagon_vertices(self, attribute_coord_x='coord_x', attribute_coord_y='coord_y',
                                list_directions_to_neighbours=None, list_directions_to_hex_vert=None):
        """
        Create the vertices of each hexagon of the graph. It requires the centroids of the hexagon to have coordinates.
        If the graph has at least one pair of neighbouring hexagon (which would be the case most of the time)

        :param attribute_coord_x:
        :param attribute_coord_y:
        :param list_directions_to_hex_vert:
        :param list_directions_to_neighbours:
        """
        X = self.df_attributes[attribute_coord_x]
        Y = self.df_attributes[attribute_coord_y]

        # if the list of directions to the hexagon vertices is provided, we just compute the vertices and exit the
        # method
        if list_directions_to_hex_vert is not None:
            self.hexagon_vertices = []
            for x, y in zip(self.df_attributes[attribute_coord_x], self.df_attributes[attribute_coord_y]):
                for u in list_directions_to_neighbours:
                    self.hexagon_vertices.append([x + u[0], y + u[1]])
            self.hexagon_vertices = np.array(self.hexagon_vertices)
            return

        # we construct, if needed, the set of all the directions toward neighbours.
        if list_directions_to_neighbours is None:
            list_directions_to_neighbours = [None for _ in range(6)]
            vect_found = False

            # to construct the directions, we need to find a vertex that has at least one neighbour. The other
            # directions are obtained by rotating it clockwise by pi/3 five times.
            for i in range(self.connections.shape[0]):
                for j in range(self.connections.shape[1]):
                    if self.connections[i, j] != -1:
                        vect_found = True
                        initial_vec = np.array([X[self.connections[i, j]] - X[i],
                                                Y[self.connections[i, j]] - Y[i]])
                        for k in range(6):
                            x = np.cos(k * np.pi / 3) * initial_vec[i][0] + \
                                np.sin(k * np.pi / 3) * initial_vec[i][1]
                            y = - np.sin(k * np.pi / 3) * initial_vec[i][0] + \
                                np.cos(k * np.pi / 3) * initial_vec[i][1]
                            list_directions_to_neighbours[(j + k) % 6] = np.array([x, y])
                        break

                # no need to check other vertex if we already found a non isolated vertex
                if vect_found:
                    break

            # if vect_found is false, it means the graph countains only isolated point (which is extremely weird).
            if not vect_found:
                raise ValueError("Could not find any hexagon with a neighbour to extrapolate the hexagons vertices. "
                                 "Use the kwarg 'list_directions_to_neighbours' to provide the neighbour direction.")

        # we know create the directions to hexagon vertices
        list_directions_to_hex_vert = []
        for i in range(6):
            list_directions_to_hex_vert.append((1/3.) * (list_directions_to_neighbours[i] +
                                                         list_directions_to_neighbours[(i + 1) % 6]))

        # finally we create the hexagon vertices
        self.hexagon_vertices = []
        for x, y in zip(self.df_attributes[attribute_coord_x], self.df_attributes[attribute_coord_y]):
            for u in list_directions_to_neighbours:
                self.hexagon_vertices.append([x + u[0], y + u[1]])
        self.hexagon_vertices = np.array(self.hexagon_vertices)


class SpatialComponentsSquareLattice:
    """

    """
    def __init__(self, **kwargs):
        pass
