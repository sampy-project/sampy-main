import numpy as np
import numba as nb
from scipy.spatial import cKDTree
from .jit_compiled_functions import (compute_max_radius,
                                     compute_area_oriented_array_of_conv_polygons_same_nb_vert,
                                     intersect_two_positively_oriented_2D_convex_polygons)


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

    def compute_intersection_same_nb_vertices(self, g1_poly_vertices=None, 
                                              attribute_g1_poly_vertices='cell_vertices',
                                              g2_poly_vertices=None, 
                                              attribute_g2_poly_vertices='cell_vertices',
                                              radius_graph_1=None, radius_graph_2=None,
                                              graph_1_coord_x='coord_x', graph_1_coord_y='coord_y',
                                              graph_2_coord_x='coord_x', graph_2_coord_y='coord_y',
                                              threshold=1e-6):
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
        :param radius_graph_1:
        :param radius_graph_2:
        :param graph_1_coord_x:
        :param graph_1_coord_y:
        :param graph_2_coord_x:
        :param graph_2_coord_y:
        :param threshold:
        """
        if g1_poly_vertices is None:
            try:
                g1_poly_vertices = getattr(self.graph_1, attribute_g1_poly_vertices)
            except AttributeError:
                raise ValueError('There is no attribute named ' + attribute_g1_poly_vertices + ' the first graph.')
        if g2_poly_vertices is None:
            try:
                g2_poly_vertices = getattr(self.graph_2, attribute_g2_poly_vertices)
            except AttributeError:
                raise ValueError('There is no attribute named ' + attribute_g2_poly_vertices + ' the second graph.')
            
        # we first create 2D arrays containing the coords of the polygons centroids
        coords_centroids_graph_1 = np.transpose(np.vstack([self.graph_1.df_attributes[graph_1_coord_x],
                                                           self.graph_1.df_attributes[graph_1_coord_y]]))
        coords_centroids_graph_2 = np.transpose(np.vstack([self.graph_2.df_attributes[graph_2_coord_x],
                                                           self.graph_2.df_attributes[graph_2_coord_y]]))
        
        # we compute the radius of the graphs if needed
        if radius_graph_1 is None:
            radius_graph_1 = compute_max_radius(coords_centroids_graph_1, g1_poly_vertices)
        if radius_graph_2 is None:
            radius_graph_2 = compute_max_radius(coords_centroids_graph_2, g2_poly_vertices)

        radius = radius_graph_1 + radius_graph_2

        # we compute the "area arrays"
        areas_graph_1 = compute_area_oriented_array_of_conv_polygons_same_nb_vert(g1_poly_vertices)
        areas_graph_2 = compute_area_oriented_array_of_conv_polygons_same_nb_vert(g2_poly_vertices)

        # we create the trees of each graph
        tree_graph_1 = cKDTree(coords_centroids_graph_1)
        tree_graph_2 = cKDTree(coords_centroids_graph_2)

        # we now compute the intersection
        list_intersect_g1_to_g2 = tree_graph_1.query_ball_tree(tree_graph_2, radius)
        
        # to compute the intersections, we will create temporary python data structure, that we will
        # later convert into proper numpy arrays
        # IMPORTANT: this whole process may be replaced by compiled code of some sort later,
        #            but it would be counter productive now
        temp_intersect_g1_to_g2 = [[] for _ in range(self.graph_1.number_vertices)]
        temp_weights_g1_to_g2 = [[] for _ in range(self.graph_1.number_vertices)]
        temp_intersect_g2_to_g1 = [[] for _ in range(self.graph_2.number_vertices)]
        temp_weights_g2_to_g1 = [[] for _ in range(self.graph_2.number_vertices)]

        for i, list_potential_intersect in enumerate(list_intersect_g1_to_g2):
            for j in list_potential_intersect:
                area = intersect_two_positively_oriented_2D_convex_polygons(g1_poly_vertices[i, :, :], g2_poly_vertices[j, :, :], threshold)
                if area > 0.:
                    temp_intersect_g1_to_g2[i].append(j)
                    temp_weights_g1_to_g2[i].append(area/areas_graph_1[i])
                    temp_intersect_g2_to_g1[j].append(i)
                    temp_weights_g2_to_g1[j].append(area/areas_graph_2[j])
