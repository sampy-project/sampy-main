import numpy as np
import numba as nb


class GraphIntersectionFromCoordsAndRadius:
    """
    This building block assumes the following attributes are here:

        graph_1: graph object
        graph_2: graph object

        indexes_g1_to_g2: 1D array of integers
        connections_g1_to_g2: 1D array of integers
        weights_g1_to_g2: 1D array of floats

        indexes_g2_to_g1: 1D array of integers
        connections_g2_to_g1: 1D array of integers
        weights_g2_to_g1: 1D array of floats
    """
    def __init__(self, **kwargs):
        pass

    def compute_intersection_2d(self, graph_1_coord_x='coord_x', graph_1_coord_y='coord_y',
                                graph_1_radius='radius', graph_2_coord_x='coord_x', graph_2_coord_y='coord_y',
                                graph_2_radius='radius'):
        """
        Compute the intersection of two 2D graphs. The intersection is computed assuming that each vertex v
        has a disk of a given radius r_v attached to it, and the intersection is just the intersection of 
        the disks.

        :param graph_1_coord_x:
        :param graph_1_coord_y:
        :param graph_1_radius:
        :param graph_2_coord_x:
        :param graph_2_coord_y:
        :param graph_2_radius:
        """
        pass

    def compute_intersection_3d(self, 
                                graph_1_coord_x='coord_x', graph_1_coord_y='coord_y', graph_1_coord_z='coord_z',
                                graph_2_coord_x='coord_x', graph_2_coord_y='coord_y', graph_2_coord_z='coord_z',
                                graph_2_radius='radius', graph_1_radius='radius'):
        """
        Compute the intersection of two 3D graphs. The intersection is computed assuming that each vertex v
        has a ball of a given radius r_v attached to it, and the inntersection is just the intersection of 
        the balls.

        :param graph_1_coord_x:
        :param graph_1_coord_y:
        :param graph_1_coord_z:
        :param graph_1_radius:
        :param graph_2_coord_x:
        :param graph_2_coord_y:
        :param graph_2_coord_z:
        :param graph_2_radius:
        """
        pass


class GraphIntersectionConvexPolygons:
    """
    This building block assumes the following attributes are here:

        graph_1: graph object
        graph_2: graph object

        indexes_g1_to_g2: 1D array of integers
        connections_g1_to_g2: 1D array of integers
        weights_g1_to_g2: 1D array of floats

        indexes_g2_to_g1: 1D array of integers
        connections_g2_to_g1: 1D array of integers
        weights_g2_to_g1: 1D array of floats
    """
    def __init__(self, **kwargs):
        pass

    def compute_intersection_same_nb_vertices(g1_poly_vertices=None, 
                                              attribute_g1_poly_vertices='cell_vertices',
                                              g2_poly_vertices=None, 
                                              attribute_g2_poly_vertices='cell_vertices'):
        """
        This method requires the following condition to be such that :
            for each graph g_j, there exists an integer n_j such that each polygon
            of g_j has exactly n_j vertices. Moreove, those vertices are supposed to be
            stored in a 3D array of shape (nb_vert_g_j, n_j, 2).
        In essence, the main use-case of this methode is to intersect two regular lattices,
        like a square lattice with an hexagonal lattice.

        :param g1_poly_vertices: 
        :param attribute_g1_poly_vertices:
        :param g2_poly_vertices:
        :param attribute_g2_poly_vertices:
        """
        if g1_poly_vertices is None:
            try:
                g1_poly_vertices = getattr(self, attribute_g1_poly_vertices)
            except AttributeError:
                raise ValueError('')