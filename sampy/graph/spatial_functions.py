import numpy as np


def create_2d_coords_from_oriented_connection_matrix(connections, index_first_vertex, coord_first_vertex,
                                                     list_vectors):
    """
    WARNING: The graph is assumed to be connected.

    Create the coordinates (2D) of each vertex of the graph. The algorithm starts at the cell given by the
        kwarg 'index_first_cell', which receives as coordinates the one given in 'coord_first_cell'. Then we loop
        through each neighbours of the starting cell, giving coordinates to each one using the parameter 'list_vector'
        (see description of 'list_vector' parameter below for a more detailed explanation). We then repeat the process
        with a vertex that has coordinates, and so on until each vertex has coordinates. This algorithm works only if
        the graph is connected.

    :param connections: 2D array of integers. Connection matrix used in SamPy Graph objects to encode the edges between
                        vertices. Here it is assumed to be oriented (that is each column correspond to a specific
                        direction).
    :param index_first_vertex: non-negative integer
    :param coord_first_vertex: couple of floats
    :param list_vectors: list containing connections.shape[1] arrays, each of shape (2,).

    :return: a pair (coords_x, coords_y) of 1D arrays of floats giving the coordinates of each vertex.
    """
    # we now create the arrays that will contain the x and y coordinates
    coord_x = np.full(connections.shape[0], 0., dtype=float)
    coord_y = np.full(connections.shape[0], 0., dtype=float)

    # we initialize the coordinates and create two data-structure that will allow us to recursively give coordinates
    # to each vertex.
    coord_x[index_first_vertex] = float(coord_first_vertex[0])
    coord_y[index_first_vertex] = float(coord_first_vertex[1])
    set_index_vert_with_coords = set([index_first_vertex])
    list_index_vert_with_coords = [index_first_vertex]

    # now we recursively give coordinates to every vertex
    for i in range(connections.shape[0]):
        try:
            current_vertex = list_index_vert_with_coords[i]
        except IndexError:
            raise ValueError("Error encountered while creating vertices' coordinates. The most likely explanation"
                             " is that the graph is not connected.")
        for j in range(connections.shape[1]):
            if connections[current_vertex, j] not in set_index_vert_with_coords:
                coord_x[connections[i, j]] = coord_x[current_vertex] + list_vectors[j][0]
                coord_y[connections[i, j]] = coord_y[current_vertex] + list_vectors[j][1]
                list_index_vert_with_coords.append(connections[i, j])
                set_index_vert_with_coords.add(connections[i, j])

    return coord_x, coord_y
