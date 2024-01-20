import numpy as np
import numba as nb


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