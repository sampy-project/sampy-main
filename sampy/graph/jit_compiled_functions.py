import numpy as np
import math
import numba as nb


@nb.njit
def topology_convert_1d_array_to_2d_array(arr_1d, arr_fast_squarify, shape_0, shape_1):
    rv = np.full((shape_0, shape_1), arr_1d[0])
    for i in range(arr_1d.shape[0]):
        rv[arr_fast_squarify[i][0]][arr_fast_squarify[i][1]] = arr_1d[i]
    return rv


@nb.njit
def topology_convert_2d_array_to_1d_array(arr_2d, arr_fast_flat):
    n = arr_2d.shape[0] * arr_2d.shape[1]
    rv = np.full((n, ), arr_2d[0][0])
    for i in range(arr_2d.shape[0]):
        for j in range(arr_2d.shape[1]):
            rv[arr_fast_flat[i][j]] = arr_2d[i][j]
    return rv


@nb.njit
def compute_sin_attr_with_condition(arr_attr, arr_cond, time, amplitude, period, phase, intercept):
    for i in range(arr_cond.shape[0]):
        if arr_cond[i]:
            arr_attr[i] = amplitude*np.sin(2*math.pi*time/period + phase) + intercept


@nb.njit
def get_oriented_neighborhood_of_vertices(connections):
    rv = np.full(connections.shape, -1, dtype=np.int32)
    for ind_center in range(connections.shape[0]):
        # we first create the set of neighbours
        set_neighbours = set()
        nb_neighbours = 0
        for i in range(connections.shape[1]):
            ind_neighb = connections[ind_center][i]
            if ind_neighb == -1:
                pass
            else:
                set_neighbours.add(ind_neighb)
                nb_neighbours += 1
        # we now fill the returned array
        for j in range(nb_neighbours):
            ind_neighbour = connections[ind_center][j]
            if ind_neighbour == -1:
                pass
            else:
                rv[ind_center][0] = ind_neighbour
                break
        for j in range(1, nb_neighbours):
            ind_current_neigh = rv[ind_center][j-1]
            for k in range(connections.shape[1]):
                ind_neighbour = connections[ind_current_neigh][k]
                if ind_neighbour == -1:
                    pass
                elif ind_neighbour in set_neighbours:
                    if j != 1 and rv[ind_center][j-2] == ind_neighbour:
                        pass
                    else:
                        rv[ind_center][j] = ind_neighbour
                        break
    return rv


@nb.njit
def get_surface_array(oriented_neighbourhood_array, x_coord, y_coord, z_coord, radius):
    rv = np.full((oriented_neighbourhood_array.shape[0],), 0., dtype=np.float64)
    for index_center in range(oriented_neighbourhood_array.shape[0]):
        # get coordinates of the center
        x_center = x_coord[index_center]
        y_center = y_coord[index_center]
        z_center = z_coord[index_center]

        # quick loop to determine the number of vertices of the current polygon
        nb_vertices = 0
        for i in range(oriented_neighbourhood_array.shape[1]):
            if oriented_neighbourhood_array[index_center][i] != -1:
                nb_vertices += 1

        # we first create the normal vectors of each hyperplane defining the spherical polygon. Those vectors are not
        # normalized
        oriented_normal_vect = np.full((nb_vertices, 3), -1.)
        current_index = 0
        for i in range(oriented_neighbourhood_array.shape[1]):
            index_current_neighbour = oriented_neighbourhood_array[index_center][i]
            if index_current_neighbour != -1:
                oriented_normal_vect[current_index][0] = x_coord[index_current_neighbour] - x_center
                oriented_normal_vect[current_index][1] = y_coord[index_current_neighbour] - y_center
                oriented_normal_vect[current_index][2] = z_coord[index_current_neighbour] - z_center
            current_index += 1

        # we know compute the coordinates of the vertices of the spherical polygon using a cross product.
        oriented_vertices_polygon = np.full((nb_vertices, 3), -1.)
        for i in range(nb_vertices):
            vertex = np.cross(oriented_normal_vect[i][:], oriented_normal_vect[(i+1) % nb_vertices][:])
            if x_center * vertex[0] + y_center * vertex[1] + z_center * vertex[2] > 0:
                oriented_vertices_polygon[i][:] = vertex / (np.sqrt((vertex ** 2).sum()))
            else:
                oriented_vertices_polygon[i][:] = - vertex / (np.sqrt((vertex ** 2).sum()))

        area = 0.
        first_point = oriented_vertices_polygon[0][:]
        second_point = oriented_vertices_polygon[1][:]
        for i in range(2, nb_vertices):
            third_point = oriented_vertices_polygon[i][:]

            vec1 = second_point - np.dot(second_point, first_point) * first_point
            vec2 = third_point - np.dot(third_point, first_point) * first_point
            area += np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

            vec1 = first_point - np.dot(first_point, second_point) * second_point
            vec2 = third_point - np.dot(third_point, second_point) * second_point
            area += np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

            vec1 = first_point - np.dot(first_point, third_point) * third_point
            vec2 = second_point - np.dot(second_point, third_point) * third_point
            area += np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

            area -= np.pi
            second_point = oriented_vertices_polygon[i][:]
        rv[index_center] = (radius**2)*area
    return rv


@nb.njit
def icosphere_get_distance_matrix(dist_matrix, connections, lats, lons, radius):
    for i in range(dist_matrix.shape[0]):
        for j in range(dist_matrix.shape[1]):
            if connections[i][j] != -1:
                dist_matrix[i][j] = radius * np.arccos(np.sin(lats[i]) * np.sin(lats[connections[i][j]]) +
                                                       np.cos(lats[i]) * np.cos(lats[connections[i][j]]) *
                                                       np.cos(lons[i] - lons[connections[i][j]]))
    return dist_matrix


@nb.njit
def keep_subgraph_from_array_of_bool_equi_weight(arr_keep, connections):
    counter = 0
    dict_old_to_new = dict()
    for i in range(arr_keep.shape[0]):
        if arr_keep[i]:
            dict_old_to_new[i] = counter
            counter += 1

    arr_nb_connections = np.full((counter,), 0, dtype=np.int32)
    new_arr_connections = np.full((counter, 6), -1, dtype=np.int32)
    new_arr_weights = np.full((counter, 6), -1., dtype=np.float32)

    counter = 0
    for i in range(arr_keep.shape[0]):
        if arr_keep[i]:
            for j in range(connections.shape[1]):
                if connections[i][j] in dict_old_to_new:
                    new_arr_connections[counter][arr_nb_connections[counter]] = dict_old_to_new[connections[i][j]]
                    arr_nb_connections[counter] += 1
            counter += 1

    for i in range(arr_nb_connections.shape[0]):
        for j in range(arr_nb_connections[i]):
            if j + 1 == arr_nb_connections[i]:
                new_arr_weights[i][j] = 1.
            else:
                new_arr_weights[i][j] = (j + 1) / arr_nb_connections[i]

    return new_arr_connections, new_arr_weights


# @nb.njit
def intersect_two_positively_oriented_2D_convex_polygons(vertices_poly_1, vertices_poly_2, threshold):
    list_vert_in_intersection = []
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

    if list_vert_in_intersection:
        center_inter = np.array([0., 0.])
        for vert in list_vert_in_intersection:
            center_inter += vert
        center_inter = center_inter / len(list_vert_in_intersection)

        list_angle_and_dist = []
        for vert in list_vert_in_intersection:
            translated_vert = vert - center_inter
            dist = np.linalg.norm(translated_vert)
            angle = np.arccos(translated_vert[0]/np.linalg.norm(translated_vert))
            oriented_angle = np.sign(-translated_vert[1]) * angle
            list_angle_and_dist.append((oriented_angle, dist))
    
        return True, np.array(list_vert_in_intersection), np.array(list_angle_and_dist)
    
    else:
        return False, None, None





    return list_vert_in_intersection

