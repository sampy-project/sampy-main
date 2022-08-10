import numpy as np

from scipy.spatial import cKDTree
from netCDF4 import Dataset

from .jit_compiled_functions import (conditional_proximity_is_step_allowed_return_infos,
                                     proximity_is_step_allowed_return_infos,
                                     proximity_get_closest_point_expand_dist_and_ind_arrays)


class ProximityDimensionThree:
    """
    todo
    """
    def __init__(self, arr_radius_point, coord_x, coord_y, coord_z):
        stacked_arr = np.column_stack([coord_x, coord_y, coord_z])
        try:
            radius = float(arr_radius_point)
            self.arr_radius_point = np.full(coord_x.shape, radius)
        except TypeError:
            self.arr_radius_point = np.array(arr_radius_point)
        self.kdtree = cKDTree(stacked_arr)

    @classmethod
    def from_graph(cls, graph, arr_radius_point, scaling_factor=1., name_df_attributes='df_attributes',
                   coord_x='coord_x', coord_y='coord_y', coord_z='coord_z'):
        df_attribute = getattr(graph, name_df_attributes)
        coord_x = scaling_factor * np.array(df_attribute[coord_x])
        coord_y = scaling_factor * np.array(df_attribute[coord_y])
        coord_z = scaling_factor * np.array(df_attribute[coord_z])
        return cls(arr_radius_point, coord_x, coord_y, coord_z)

    @classmethod
    def from_netcdf(cls, path_to_netcdf, name_lats_attr, name_lons_attr, arr_radius_point=None, radius=1.):
        """
        todo
        :param path_to_netcdf:
        :param name_lats_attr:
        :param name_lons_attr:
        :param arr_radius_point:
        :param radius:
        :return:
        """
        data = Dataset(path_to_netcdf, mode='r')
        nf_lats = np.array(data.variables[name_lats_attr][:])
        nf_lons = np.array(data.variables[name_lons_attr][:])
        lats = nf_lats.flatten()
        lons = nf_lons.flatten()
        coord_x = radius * np.cos(np.radians(lats)) * np.cos(np.radians(lons))
        coord_y = radius * np.cos(np.radians(lats)) * np.sin(np.radians(lons))
        coord_z = radius * np.sin(np.radians(lats))
        if arr_radius_point is None:
            nf_coord_x = radius * np.cos(np.radians(nf_lats)) * np.cos(np.radians(nf_lons))
            nf_coord_y = radius * np.cos(np.radians(nf_lats)) * np.sin(np.radians(nf_lons))
            nf_coord_z = radius * np.sin(np.radians(nf_lats))
            arr_radius_point = np.full(nf_lats.shape, -1.)
            for i in range(nf_lats.shape[0]):
                for j in range(nf_lats.shape[1]):
                    current_max_dist = 0.
                    if i-1 >= 0:
                        d = (nf_coord_x[i-1, j] - nf_coord_x[i, j])**2 + \
                            (nf_coord_y[i-1, j] - nf_coord_y[i, j])**2 + \
                            (nf_coord_z[i-1, j] - nf_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    if j-1 >= 0:
                        d = (nf_coord_x[i, j-1] - nf_coord_x[i, j])**2 + \
                            (nf_coord_y[i, j-1] - nf_coord_y[i, j])**2 + \
                            (nf_coord_z[i, j-1] - nf_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    try:
                        d = (nf_coord_x[i, j + 1] - nf_coord_x[i, j]) ** 2 + \
                            (nf_coord_y[i, j + 1] - nf_coord_y[i, j]) ** 2 + \
                            (nf_coord_z[i, j + 1] - nf_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    except IndexError:
                        pass
                    try:
                        d = (nf_coord_x[i+1, j] - nf_coord_x[i, j]) ** 2 + \
                            (nf_coord_y[i+1, j] - nf_coord_y[i, j]) ** 2 + \
                            (nf_coord_z[i+1, j] - nf_coord_z[i, j]) ** 2
                        d = np.sqrt(d)
                        if d > current_max_dist:
                            current_max_dist = d
                    except IndexError:
                        pass
                    arr_radius_point[i, j] = current_max_dist
            arr_radius_point = arr_radius_point.flatten()
        r_cls = cls(arr_radius_point, coord_x, coord_y, coord_z)
        setattr(r_cls, 'netcdf', data)
        return r_cls

    def get_closest_point(self, arr_target_x, arr_target_y, arr_target_z, selected_agents=None):
        """
        todo
        :param arr_target_x:
        :param arr_target_y:
        :param arr_target_z:
        :param selected_agents: Optional, array of bool, default None. If the value is not None, only the agents for
            which the value in this array is True will be considered.
        :return: couple of arrays (distance, indices), the second one giving the index of the closest point and the
            first one the distance to this point
        """
        if selected_agents is None:
            t_x = np.array(arr_target_x)
            t_y = np.array(arr_target_y)
            t_z = np.array(arr_target_z)
            stacked_arr = np.column_stack([t_x, t_y, t_z])
            return self.kdtree.query(stacked_arr, k=1)
        else:
            selected_agents = np.array(selected_agents)
            t_x = np.array(arr_target_x[selected_agents])
            t_y = np.array(arr_target_y[selected_agents])
            t_z = np.array(arr_target_z[selected_agents])
            stacked_arr = np.column_stack([t_x, t_y, t_z])
            # print(stacked_arr)
            distances, indices = self.kdtree.query(stacked_arr, k=1)
            # print(indices)
            return proximity_get_closest_point_expand_dist_and_ind_arrays(selected_agents, distances, indices)

    def is_step_allowed(self, arr_selected_agents, pos_x, pos_y, pos_z, condition_on_grid=None):
        """
        todo
        :param arr_selected_agents:
        :param pos_x:
        :param pos_y:
        :param pos_z:
        :param condition_on_grid: Optional, array of bool, default None.
        :return: three arrays, first one of bool, second of distance, third of indexes.
        """
        distances, indices = self.get_closest_point(pos_x, pos_y, pos_z, selected_agents=arr_selected_agents)
        # print(indices)
        # print(arr_selected_agents)
        # print(' ')
        if condition_on_grid is not None:
            condition_on_grid = np.array(condition_on_grid)
            return conditional_proximity_is_step_allowed_return_infos(arr_selected_agents, distances, indices,
                                                                      self.arr_radius_point, condition_on_grid)
        else:
            return proximity_is_step_allowed_return_infos(arr_selected_agents, distances, indices,
                                                          self.arr_radius_point)
