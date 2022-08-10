import numpy as np
import numba as nb


@nb.njit
def conditional_proximity_is_step_allowed_return_infos(arr_selected_agents, distances, indices, arr_radius_point,
                                                       condition_on_grid):
    rv = np.full(arr_selected_agents.shape, False, dtype=np.bool_)
    r_d = np.full(arr_selected_agents.shape, -1., dtype=np.float_)
    r_ind = np.full(arr_selected_agents.shape, -1, dtype=np.int32)
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            if condition_on_grid[indices[i]] and (distances[i] <= arr_radius_point[indices[i]]):
                rv[i] = True
                r_d[i] = distances[i]
                r_ind[i] = indices[i]
    return rv, r_d, r_ind


@nb.njit
def proximity_is_step_allowed_return_infos(arr_selected_agents, distances, indices, arr_radius_point):
    rv = np.full(arr_selected_agents.shape, False, dtype=np.bool_)
    r_d = np.full(arr_selected_agents.shape, -1., dtype=np.float_)
    r_ind = np.full(arr_selected_agents.shape, -1, dtype=np.int32)
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            if distances[i] <= arr_radius_point[indices[i]]:
                rv[i] = True
                r_d[i] = distances[i]
                r_ind[i] = indices[i]
    return rv, r_d, r_ind


@nb.njit
def proximity_get_closest_point_expand_dist_and_ind_arrays(selected_agents, distances, indexes):
    r_d = np.full(selected_agents.shape, -1., dtype=np.float_)
    r_ind = np.full(selected_agents.shape, -1, dtype=np.int32)
    counter = 0
    for i in range(selected_agents.shape[0]):
        if selected_agents[i]:
            r_d[i] = distances[counter]
            r_ind[i] = indexes[counter]
            counter += 1
    return r_d, r_ind
