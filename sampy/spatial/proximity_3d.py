import numpy as np

from scipy.spatial import cKDTree
# from netCDF4 import Dataset

from .jit_compiled_functions import (conditional_proximity_is_step_allowed_return_infos,
                                     proximity_is_step_allowed_return_infos,
                                     conditional_proximity_is_pos_allowed,
                                     proximity_is_pos_allowed)


class BaseProximity3dFromArrays:
    """
    Base class for constructing 3d-proximity object from raw one dimensional arrays.

    :param arr_radius_point: 1d array of float, giving the "radius" around each point of the proximity object.
    :param coord_x: 1d array of floats, x coordinates of points
    :param coord_y: 1d array of floats, y coordinates of points
    :param coord_z: 1d array of floats, z coordinates of points
    :param allowed_points: optional, 1D array of bool, default None.
    """
    def __init__(self, arr_radius_point=None, coord_x=None, coord_y=None, coord_z=None, allowed_points=None, **kwargs):
        if arr_radius_point is None:
            raise ValueError("No value for arr_radius_point kwarg has been given.")
        if coord_x is None:
            raise ValueError("No value for coord_x kwarg has been given.")
        if coord_y is None:
            raise ValueError("No value for coord_y kwarg has been given.")
        if coord_z is None:
            raise ValueError("No value for coord_z kwarg has been given.")

        stacked_arr = np.column_stack([coord_x, coord_y, coord_z])
        try:
            radius = float(arr_radius_point)
            self.arr_radius_point = np.full(coord_x.shape, radius)
        except TypeError:
            self.arr_radius_point = np.copy(arr_radius_point)

        self.coord_x = np.copy(coord_x)
        self.coord_y = np.copy(coord_y)
        self.coord_z = np.copy(coord_z)
        self.kdtree = cKDTree(stacked_arr)

        self.allowed_points = allowed_points

    def update_allowed_points(self, arr_new_allowed_point):
        """
        Update the array of allowed points.

        :param arr_new_allowed_point: 1D array of bool, telling which point is allowed.
        """
        self.allowed_points = arr_new_allowed_point


class BaseProximity3dFromLatLonGrid:
    """
    Base class for constructing 3d-proximity object from two 2d grids of lattitude and longitude.

    :param grid_lats: 2D array of float, latitudes of the points.
    :param grid_lons: 2D array of float, longitude of the points.
    :param radius: float, radius of the sphere on which the points live
    :param arr_radius_point: 1d array of float, giving the "radius" around each point of the proximity object.
    :param allowed_points: optional, 2D array of bool, default None.
    """
    def __init__(self, grid_lats=None, grid_lons=None, radius=None, arr_radius_point=None, allowed_points=None,
                 **kwargs):
        if grid_lats is None:
            raise ValueError("No value for grid_lats kwarg has been given.")
        if grid_lons is None:
            raise ValueError("No value for grid_lons kwarg has been given.")
        if radius is None:
            raise ValueError("No value for radius kwarg has been given.")

        # start creating the proximity 3d thingy
        grid_coord_x = radius * np.cos(np.radians(grid_lats)) * np.cos(np.radians(grid_lons))
        grid_coord_y = radius * np.cos(np.radians(grid_lats)) * np.sin(np.radians(grid_lons))
        grid_coord_z = radius * np.sin(np.radians(grid_lats))

        if arr_radius_point is None:
            arr_radius_point = np.full(grid_lats.shape, -1.)
            for i in range(grid_lats.shape[0]):
                for j in range(grid_lats.shape[1]):
                    current_max_dist = 0.
                    if i - 1 >= 0:
                        d = (grid_coord_x[i - 1, j] - grid_coord_x[i, j]) ** 2 + \
                            (grid_coord_y[i - 1, j] - grid_coord_y[i, j]) ** 2 + \
                            (grid_coord_z[i - 1, j] - grid_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    if j - 1 >= 0:
                        d = (grid_coord_x[i, j - 1] - grid_coord_x[i, j]) ** 2 + \
                            (grid_coord_y[i, j - 1] - grid_coord_y[i, j]) ** 2 + \
                            (grid_coord_z[i, j - 1] - grid_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    try:
                        d = (grid_coord_x[i, j + 1] - grid_coord_x[i, j]) ** 2 + \
                            (grid_coord_y[i, j + 1] - grid_coord_y[i, j]) ** 2 + \
                            (grid_coord_z[i, j + 1] - grid_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    except IndexError:
                        pass
                    try:
                        d = (grid_coord_x[i + 1, j] - grid_coord_x[i, j]) ** 2 + \
                            (grid_coord_y[i + 1, j] - grid_coord_y[i, j]) ** 2 + \
                            (grid_coord_z[i + 1, j] - grid_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    except IndexError:
                        pass
                    arr_radius_point[i, j] = current_max_dist

        self.coord_x = grid_coord_x.flatten()
        self.coord_y = grid_coord_y.flatten()
        self.coord_z = grid_coord_z.flatten()
        self.arr_radius_point = arr_radius_point.flatten()
        self.kdtree = cKDTree(np.column_stack([self.coord_x, self.coord_y, self.coord_z]))
        self.allowed_points = allowed_points

    def update_allowed_points(self, arr_new_allowed_point):
        """
        Update the array of allowed points.

        :param arr_new_allowed_point: 2D array of bool, telling which point of the grid is allowed.
        """
        self.allowed_points = arr_new_allowed_point.flatten()


class Proximity3dBasicSpatialQueries:
    def __init__(self, **kwargs):
        pass

    def get_closest_point(self, arr_target_x, arr_target_y, arr_target_z):
        """
        Given a series of target-point whose coordinates are given as three 1 dimensional arrays, this method gives the
        index of the closest point (and the distance) in the proximity class to each of the target-points.

        :param arr_target_x: 1d array of float, x coordinate of target point.
        :param arr_target_y: 1d array of float, y coordinate of target point.
        :param arr_target_z: 1d array of float, z coordinate of target point.

        :return: couple of arrays (distance, indices), the second one giving the index of the closest point and the
            first one the distance to this point
        """
        stacked_arr = np.column_stack([arr_target_x, arr_target_y, arr_target_z])
        return self.kdtree.query(stacked_arr, k=1)

    def is_pos_allowed(self, pos_x, pos_y, pos_z, distances=None, indices=None):
        """
        Given a series of position, whose coordinates are given as three 1 dimensional arrays, this method tells which
        are allowed with respect to the current proximity class.

        :param pos_x: 1d array of float, x coordinate of the positions.
        :param pos_y: 1d array of float, y coordinate of the positions
        :param pos_z: 1d array of float, z coordinate of the positions
        :param distances: optional, 1d array of float, default None. If given, distances[i] should be the distance
                          from the position i to its closest point in the proximity class.
        :param indices: optional, 1d array of int, default None. If given, indices[i] should be the index of the point
                        in the proximity class which is the closest to the position i.

        :return: an array of bool, telling which positions are allowed.
        """
        if (indices is None) or (distances is None):
            distances, indices = self.get_closest_point(pos_x, pos_y, pos_z)
        if self.allowed_points is not None:
            return conditional_proximity_is_pos_allowed(indices, distances, self.arr_radius_point, self.allowed_points)
        else:
            return proximity_is_pos_allowed(indices, distances, self.arr_radius_point)


# class LegacyNetcdfThingy:
#     def __init__(self, **kwargs):
#         pass
#
#     @classmethod
#     def from_netcdf(cls, path_to_netcdf, name_lats_attr, name_lons_attr, arr_radius_point=None, radius=1.):
#         """
#         todo
#         :param path_to_netcdf:
#         :param name_lats_attr:
#         :param name_lons_attr:
#         :param arr_radius_point:
#         :param radius:
#         :return:
#         """
#         data = Dataset(path_to_netcdf, mode='r')
#         nf_lats = np.array(data.variables[name_lats_attr][:])
#         nf_lons = np.array(data.variables[name_lons_attr][:])
#         lats = nf_lats.flatten()
#         lons = nf_lons.flatten()
#         coord_x = radius * np.cos(np.radians(lats)) * np.cos(np.radians(lons))
#         coord_y = radius * np.cos(np.radians(lats)) * np.sin(np.radians(lons))
#         coord_z = radius * np.sin(np.radians(lats))
#         if arr_radius_point is None:
#             nf_coord_x = radius * np.cos(np.radians(nf_lats)) * np.cos(np.radians(nf_lons))
#             nf_coord_y = radius * np.cos(np.radians(nf_lats)) * np.sin(np.radians(nf_lons))
#             nf_coord_z = radius * np.sin(np.radians(nf_lats))
#             arr_radius_point = np.full(nf_lats.shape, -1.)
#             for i in range(nf_lats.shape[0]):
#                 for j in range(nf_lats.shape[1]):
#                     current_max_dist = 0.
#                     if i-1 >= 0:
#                         d = (nf_coord_x[i-1, j] - nf_coord_x[i, j])**2 + \
#                             (nf_coord_y[i-1, j] - nf_coord_y[i, j])**2 + \
#                             (nf_coord_z[i-1, j] - nf_coord_z[i, j]) ** 2
#                         d = np.sqrt(d)
#                         if d > current_max_dist:
#                             current_max_dist = d
#                     if j-1 >= 0:
#                         d = (nf_coord_x[i, j-1] - nf_coord_x[i, j])**2 + \
#                             (nf_coord_y[i, j-1] - nf_coord_y[i, j])**2 + \
#                             (nf_coord_z[i, j-1] - nf_coord_z[i, j]) ** 2
#                         d = np.sqrt(d)
#                         if d > current_max_dist:
#                             current_max_dist = d
#                     try:
#                         d = (nf_coord_x[i, j + 1] - nf_coord_x[i, j]) ** 2 + \
#                             (nf_coord_y[i, j + 1] - nf_coord_y[i, j]) ** 2 + \
#                             (nf_coord_z[i, j + 1] - nf_coord_z[i, j]) ** 2
#                         d = np.sqrt(d)
#                         if d > current_max_dist:
#                             current_max_dist = d
#                     except IndexError:
#                         pass
#                     try:
#                         d = (nf_coord_x[i+1, j] - nf_coord_x[i, j]) ** 2 + \
#                             (nf_coord_y[i+1, j] - nf_coord_y[i, j]) ** 2 + \
#                             (nf_coord_z[i+1, j] - nf_coord_z[i, j]) ** 2
#                         d = np.sqrt(d)
#                         if d > current_max_dist:
#                             current_max_dist = d
#                     except IndexError:
#                         pass
#                     arr_radius_point[i, j] = current_max_dist
#             arr_radius_point = arr_radius_point.flatten()
#         r_cls = cls(arr_radius_point, coord_x, coord_y, coord_z)
#         setattr(r_cls, 'netcdf', data)
#         return r_cls
