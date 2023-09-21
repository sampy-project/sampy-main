import numpy as np
import numba as nb
from numba.typed import List


@nb.njit
def conditional_proximity_is_step_allowed_return_infos(arr_selected_agents, distances, indices, arr_radius_point,
                                                       condition_on_grid):
    rv = np.full(arr_selected_agents.shape, False, dtype=np.bool_)
    r_d = np.full(arr_selected_agents.shape, -1., dtype=float)
    r_ind = np.full(arr_selected_agents.shape, -1, dtype=np.int32)
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            if condition_on_grid[indices[i]] and (distances[i] <= arr_radius_point[indices[i]]):
                rv[i] = True
                r_d[i] = distances[i]
                r_ind[i] = indices[i]
    return rv, r_d, r_ind


@nb.njit
def conditional_proximity_is_pos_allowed(indices, distances, arr_radius_points, allowed_points):
    rv = np.full(indices.shape, False, dtype=np.bool_)
    for i in range(indices.dhape[0]):
        if allowed_points[indices[i]] and (distances[i] <= arr_radius_points[indices[i]]):
            rv = True
    return rv


@nb.njit
def proximity_is_pos_allowed(indices, distances, arr_radius_points):
    rv = np.full(indices.shape, False, dtype=np.bool_)
    for i in range(indices.dhape[0]):
        if distances[i] <= arr_radius_points[indices[i]]:
            rv = True
    return rv


@nb.njit
def proximity_is_step_allowed_return_infos(arr_selected_agents, distances, indices, arr_radius_point):
    rv = np.full(arr_selected_agents.shape, False, dtype=np.bool_)
    r_d = np.full(arr_selected_agents.shape, -1., dtype=float)
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
    r_d = np.full(selected_agents.shape, -1., dtype=float)
    r_ind = np.full(selected_agents.shape, -1, dtype=np.int32)
    counter = 0
    for i in range(selected_agents.shape[0]):
        if selected_agents[i]:
            r_d[i] = distances[counter]
            r_ind[i] = indexes[counter]
            counter += 1
    return r_d, r_ind


@nb.njit
def intersect_two_positively_oriented_2D_convex_polygons(vertices_poly_1, vertices_poly_2, threshold):
    list_vert_in_intersection = List()
    for i in range(vertices_poly_2.shape[0]):
        is_in_poly_1 = True
        for j in range(vertices_poly_1.shape[0]):
            next_ind = (j + 1) % vertices_poly_1.shape[0]
            prod_scal = (vertices_poly_2[i, 0] - vertices_poly_1[j, 0])*(vertices_poly_1[j, 1] - vertices_poly_1[next_ind, 1]) + \
                        (vertices_poly_2[i, 1] - vertices_poly_1[j, 1])*(vertices_poly_1[next_ind, 0] - vertices_poly_1[j, 0])
            if prod_scal < - threshold:
                is_in_poly_1 = False
                break
        if is_in_poly_1:
            list_vert_in_intersection.append(vertices_poly_2[i, :])
    
    for i in range(vertices_poly_1.shape[0]):
        is_in_poly_2 = True
        for j in range(vertices_poly_2.shape[0]):
            next_ind = (j + 1) % vertices_poly_2.shape[0]
            prod_scal = (vertices_poly_1[i, 0] - vertices_poly_2[j, 0])*(vertices_poly_2[j, 1] - vertices_poly_2[next_ind, 1]) + \
                        (vertices_poly_1[i, 1] - vertices_poly_2[j, 1])*(vertices_poly_2[next_ind, 0] - vertices_poly_2[j, 0])
            if prod_scal < - threshold:
                is_in_poly_2 = False
                break
        if is_in_poly_2:
            list_vert_in_intersection.append(vertices_poly_1[i, :])

    # we now look for the intersections between the sides of the polygons
    # we follow the method and notations used in this post :
    # https://stackoverflow.com/questions/563198/how-do-you-detect-where-two-line-segments-intersect
    for i in range(vertices_poly_1.shape[0]):
        next_i = (i + 1) % vertices_poly_1.shape[0]
        for j in range(vertices_poly_2.shape[0]):
            next_j = (j + 1) % vertices_poly_2.shape[0]
            r = vertices_poly_1[next_i, :] - vertices_poly_1[i, :]
            s = vertices_poly_2[next_j, :] - vertices_poly_2[j, :]
            r_cross_s = r[0] * s[1] - r[1] * s[0]

            # if r_cross_s is 0, then the segments are parallel
            if np.abs(r_cross_s) < threshold:
                continue

            q_minus_p = vertices_poly_2[j, :] - vertices_poly_1[i, :]
            t = (q_minus_p[0] * s[1] - q_minus_p[1] * s[0]) / r_cross_s

            # if t is negative or bigger than 1, then the lines intersect, but not the segments
            if t < 0 or t > 1:
                continue

            s = (q_minus_p[0] * r[1] - q_minus_p[1] * r[0]) / r_cross_s

            # same remark as before
            if s < 0 or s > 1:
                continue

            list_vert_in_intersection.append(vertices_poly_1[i, :] + t * r)

    # if len(list_vert_in_intersection) <= 2:
    #     return 0.

    # we get rid of the repetitions using the threshold value
    arr_keep_vertices = np.full(len(list_vert_in_intersection), True)
    for i in range(len(list_vert_in_intersection)):
        for j in range(i + 1, len(list_vert_in_intersection)):
            if arr_keep_vertices[j] and \
                np.linalg.norm(list_vert_in_intersection[i] - list_vert_in_intersection[j]) < threshold:
                arr_keep_vertices[j] = False

    vert_in_intersection = np.full((arr_keep_vertices.sum(), 2), 0.)
    counter = 0
    for i in range(len(list_vert_in_intersection)):
        if arr_keep_vertices[i]:
            vert_in_intersection[counter, 0] = list_vert_in_intersection[i][0]
            vert_in_intersection[counter, 1] = list_vert_in_intersection[i][1]
            counter += 1

    # we compute oriented angles to order the vertices in the direct orientation 
    # (needed for computing the area)
    arr_angles = np.full(vert_in_intersection.shape[0], 0.)
    center_inter = np.array([0., 0.])
    for i in range(vert_in_intersection.shape[0]):
        center_inter += vert_in_intersection[i, :]
    center_inter = center_inter / vert_in_intersection.shape[0]

    for i in range(arr_angles.shape[0]):
        translated_vert = vert_in_intersection[i] - center_inter
        angle = np.arccos(translated_vert[0]/np.linalg.norm(translated_vert))
        oriented_angle = np.sign(translated_vert[1]) * angle
        arr_angles[i] = oriented_angle

    vert_in_intersection = vert_in_intersection[np.argsort(arr_angles)]

    area = 0.
    for i in range(vert_in_intersection.shape[0]):
        next_i = (i + 1) % vert_in_intersection.shape[0]
        area += vert_in_intersection[i, 0] * vert_in_intersection[next_i, 1] - \
            vert_in_intersection[i, 1] * vert_in_intersection[next_i, 0]
    area = area / 2.

    return area


@nb.njit
def compute_area_oriented_array_of_conv_polygons_same_nb_vert(array_vertices):
    array_area = np.zeros((array_vertices.shape[0],), dtype=float)
    for i in range(array_vertices.shape[0]):
        for j in range(array_vertices.shape[1]): # that's where the assumption that the number
                                                 # of vertices is the same accross all polyg
            next_j = (j + 1) % array_vertices.shape[1]
            array_area[i] += array_vertices[i, j, 0] * array_vertices[i, next_j, 1] - \
                array_vertices[i, j, 1] * array_vertices[i, next_j, 0]
        array_area[i] /= 2.
    return array_area


@nb.njit
def compute_max_radius(centroids, polygon_vertices):
    current_radius = 0.
    for i in range(centroids.shape[0]):
        for j in range(polygon_vertices.shape[1]):
            new_radius = np.sqrt((centroids[i, 0] - polygon_vertices[i, j, 0]) ** 2 + (centroids[i, 1] - polygon_vertices[i, j, 1]) ** 2)
            if new_radius > current_radius:
                current_radius = new_radius
    return current_radius