from .misc import (create_grid_hexagonal_cells,
                   create_grid_square_cells,
                   create_grid_square_with_diagonals,
                   SubdividedIcosahedron)
from .jit_compiled_functions import (get_oriented_neighborhood_of_vertices,
                                     get_surface_array,
                                     topology_convert_1d_array_to_2d_array,
                                     topology_convert_2d_array_to_1d_array,
                                     icosphere_get_distance_matrix)


import numpy as np
from math import sqrt, pi
import pandas as pd

import os


class BaseTopology:
    def __init__(self, **kwargs):
        self.connections = None
        self.weights = None
        self.type = None
        self.dict_cell_id_to_ind = {}
        self.time = 0

        self.on_ticker = ['increment_time']

    def increment_time(self):
        self.time += 1

    def tick(self):
        """
        execute the methods whose names are stored in the attribute on_ticker, in order.
        """
        for method in self.on_ticker:
            getattr(self, method)()

    def save_table_id_of_vertices_to_indices(self, path_to_csv, sep, erase_existing_file=True):
        """
        Create and save a two column csv allowing to match vertices id's with vertices indexes.

        :param path_to_csv: string, path to the output csv
        :param sep: string, separator to use in the csv.
        :param erase_existing_file: optional, boolean, default True. If True, the method will check if there is already
                                    a file at path_to_csv and delete it if it exists.
        """
        if erase_existing_file:
            if os.path.exists(path_to_csv):
                os.remove(path_to_csv)
        with open(path_to_csv, 'a') as f_out:
            f_out.write("id_vertex" + sep + "index_vertex" + "\n")
            for id_vertex, index in self.dict_cell_id_to_ind.items():
                f_out.write(str(id_vertex) + sep + str(index) + '\n')
        return

    @property
    def number_vertices(self):
        return self.connections.shape[0]


class SquareGridWithDiagTopology(BaseTopology):
    def __init__(self, shape=None, **kwargs):
        if shape is None:
            raise ValueError("Kwarg 'shape' is missing while initializing the graph topology. 'shape' should be a "
                             "tuple like object of the form (a, b), where a and b are integers bigger than 1.")
        len_side_a = shape[0]
        len_side_b = shape[1]
        self.create_square_with_diag_grid(len_side_a, len_side_b)
        self.shape = (len_side_a, len_side_b)
        self.type = 'SquareGridWithDiag'

    def create_square_with_diag_grid(self, len_side_a, len_side_b):
        """
        Create a square grid with diagonals, where each vertex X[i][j] is linked to X[i-1][j-1], X[i][j-1], X[i+1][j-1],
        X[i+1][j], X[i+1][j+1], x[i][j+1], x[i-1][j+1] and x[i-1][j] if they exist. Note that the weights on the
        'diagonal connections' is reduced to take into account the fact that the vertices on the diagonal are 'further
        away' (i.e. using sqrt(2) as a distance instead of 1 in the weight computation).

        :param len_side_a: integer, x coordinate
        :param len_side_b: integer, y coordinate
        """
        if (len_side_a < 2) or (len_side_b < 2):
            raise ValueError('side length attributes for HexagonalCells should be at least 2.')

        self.connections, self.weights = create_grid_square_with_diagonals(len_side_a, len_side_b)

        # populate the dictionary from cell coordinates to cell indexes in arrays connection and weights
        for i in range(len_side_a):
            for j in range(len_side_b):
                self.dict_cell_id_to_ind[(i, j)] = j + i*len_side_b


class SquareGridTopology(BaseTopology):
    def __init__(self, shape=None, **kwargs):
        if shape is None:
            raise ValueError("Kwarg 'shape' is missing while initializing the graph topology. 'shape' should be a "
                             "tuple like object of the form (a, b), where a and b are integers bigger than 1.")
        len_side_a = shape[0]
        len_side_b = shape[1]
        self.create_square_grid(len_side_a, len_side_b)
        self.shape = (len_side_a, len_side_b)
        self.type = 'SquareGrid'

    def create_square_grid(self, len_side_a, len_side_b):
        """
        Create a square grid, where each vertex X[i][j] is linked to X[i-1][j], X[i][j-1], X[i+1][j], X[i][j+1] if they
        exist.

        :param len_side_a: integer, x coordinate
        :param len_side_b: integer, y coordinate
        """
        if (len_side_a < 2) or (len_side_b < 2):
            raise ValueError('side length attributes for HexagonalCells should be at least 2.')

        self.connections, self.weights = create_grid_square_cells(len_side_a, len_side_b)
        self.shape = (len_side_a, len_side_b)

        # populate the dictionary from cell coordinates to cell indexes in arrays connection and weights
        for i in range(len_side_a):
            for j in range(len_side_b):
                self.dict_cell_id_to_ind[(i, j)] = j + i*len_side_b


class SquareGridsConvertBetween1DArrayAnd2DArrays:
    def __init__(self, **kwargs):
        self._array_optimized_squarification = None
        self._array_optimized_flat = None

    def _sampy_debug_convert_1d_array_to_2d_array(self, input_arr):
        if not type(input_arr) is np.ndarray:
            raise ValueError("Input variable is not a numpy array.")
        if input_arr.shape != (self.number_vertices,):
            raise ValueError("Input array of invalid shape.")

    def convert_1d_array_to_2d_array(self, input_arr):
        """
        Takes a 1D array of shape (nb_vertices,) and convert it into a 2D array of shape self.shape.

        :param input_arr: 1D array.
        :return: 2D array
        """
        if self._array_optimized_squarification is None:
            self._array_optimized_squarification = np.full((self.number_vertices, 2), 0)
            for key, val in self.dict_cell_id_to_ind.items():
                self._array_optimized_squarification[val][0] = key[0]
                self._array_optimized_squarification[val][1] = key[1]
        return topology_convert_1d_array_to_2d_array(input_arr, self._array_optimized_squarification,
                                                     self.shape[0], self.shape[1])

    def _sampy_debug_convert_2d_array_to_1d_array(self, input_arr):
        if not type(input_arr) is np.ndarray:
            raise ValueError("Input variable is not a numpy array.")
        if input_arr.shape != self.shape:
            raise ValueError("Input array of invalid shape.")

    def convert_2d_array_to_1d_array(self, input_array):
        """

        :param input_array:
        :return:
        """
        if self._array_optimized_flat is None:
            self._array_optimized_flat = np.full(self.shape, 0)
            for key, val in self.dict_cell_id_to_ind.items():
                self._array_optimized_flat[key[0], key[1]] = val
        return topology_convert_2d_array_to_1d_array(input_array, self._array_optimized_flat)


class IcosphereTopology(BaseTopology):
    def __init__(self, nb_sub=None, radius=1., **kwargs):
        if nb_sub is None:
            raise ValueError("kwarg nb_sub missing")
        self.nb_sub = nb_sub
        self.radius = float(radius)

        icosahedron = SubdividedIcosahedron(nb_sub)
        self.connections = np.copy(icosahedron.connections)
        self.weights = np.copy(icosahedron.weights)
        self.arr_coord = np.copy(icosahedron.arr_coord)
        del icosahedron

        self.type = 'IcoSphere'
        self.three_d_coord_created = False

    def create_3d_coord(self):
        self.df_attributes['coord_x'] = self.arr_coord[:, 0].astype(np.float64)
        self.df_attributes['coord_y'] = self.arr_coord[:, 1].astype(np.float64)
        self.df_attributes['coord_z'] = self.arr_coord[:, 2].astype(np.float64)

        norm = np.sqrt(self.df_attributes['coord_x']**2 +
                       self.df_attributes['coord_y']**2 +
                       self.df_attributes['coord_z']**2)

        self.df_attributes['coord_x_normalized'] = self.df_attributes['coord_x'] / norm
        self.df_attributes['coord_x'] = self.radius * self.df_attributes['coord_x_normalized']

        self.df_attributes['coord_y_normalized'] = self.df_attributes['coord_y'] / norm
        self.df_attributes['coord_y'] = self.radius * self.df_attributes['coord_y_normalized']

        self.df_attributes['coord_z_normalized'] = self.df_attributes['coord_z'] / norm
        self.df_attributes['coord_z'] = self.radius * self.df_attributes['coord_z_normalized']

        self.three_d_coord_created = True

    def create_pseudo_epsg4326_coordinates(self):
        """
        This method approximate the shape of the earth using a sphere, which creates deformations.
        """
        if not self.three_d_coord_created:
            self.create_3d_coord()

        self.df_attributes['lat'] = (180*(pi/2 - np.arccos(self.df_attributes['coord_z_normalized']))/pi).astype(np.float64)
        self.df_attributes['lon'] = (180*np.arctan2(self.df_attributes['coord_y_normalized'],
                                                   self.df_attributes['coord_x_normalized'])/pi).astype(np.float64)

    def compute_distance_matrix_on_sphere(self):
        """
        This method compute a distance matrix that gives the distance between each pair of connected vertex of the
        graph. The distance is the geodesic distance on a sphere (i.e. the distance

        :return: Array of floats with the same shape as the array 'connections'
        """
        dist_matrix = np.full(self.connections.shape, -1., dtype=np.float64)
        lats_rad = (np.pi * self.df_attributes['lat'] / 180).astype(np.float64)
        lons_rad = (np.pi * self.df_attributes['lon'] / 180).astype(np.float64)

        dist_matrix = icosphere_get_distance_matrix(dist_matrix, self.connections, lats_rad, lons_rad, self.radius)

        return dist_matrix

    def create_and_save_radius_cells_as_attribute(self, radius_attribute='radius_each_cell'):
        """
        Save radius of each cell. The radius of a cell centered on a vertex v is defined as the maximum distance between
        v and its neighbours. The radius is saved within df_attributes.

        :param radius_attribute: optional, string, default 'radius_each_cell'. Name of the attribute corresponding to
                                 the radius of each cell.
        """
        dist_matrix = self.compute_distance_matrix_on_sphere()
        max_distance = np.amax(dist_matrix, axis=1).astype(np.float64)
        self.df_attributes[radius_attribute] = max_distance

    def compute_surface_array(self):
        """
        Return an array giving the surface of each cell of the icosphere
        :return: array of floats shape (nb_vertex,)
        """
        # note that this orientation is not necessarily clockwise, and can be anti-clockwise for some vertices.
        # But this is not important for our purpose of computing the area of each cell of the icosphere.
        oriented_neigh_vert = get_oriented_neighborhood_of_vertices(self.connections)

        return get_surface_array(oriented_neigh_vert,
                                 self.df_attributes['coord_x_normalized'],
                                 self.df_attributes['coord_y_normalized'],
                                 self.df_attributes['coord_z_normalized'],
                                 self.radius).astype(np.float64)

    def create_and_save_surface_array_as_attribute(self):
        arr_surface = self.compute_surface_array()
        self.df_attributes['surface_cell'] = arr_surface


class OrientedHexagonalGrid(BaseTopology):
    """
    Create an hexagonal lattice on a square. Each
    """
    def __init__(self, nb_hex_x_axis=None, nb_hex_y_axis=None, **kwargs):
        """
        :param nb_hex_x_axis: mandatory kwargs. Integer, number of hexagons on the horizontal axis.
        :param nb_hex_y_axis: mandatory kwargs. Integer, number of hexagons on the vertical axis.
        """
        pass