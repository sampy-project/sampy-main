import numba as nb
import numpy as np
from numba.typed import List


# ----------------------------------------------------------------------------------------------------------------------
# generic functions
@nb.njit
def conditional_count_return_full_array(nb_vertex, arr_pos, arr_cond):
    pos_count = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_cond[i]:
            pos_count[arr_pos[i]] += 1
    rv = np.zeros((arr_pos.shape[0],), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        rv[i] = pos_count[arr_pos[i]]
    return rv

# ----------------------------------------------------------------------------------------------------------------------
# base


@nb.njit
def base_conditional_count_nb_agent_per_vertex(arr_condition, arr_pos, nb_vertex):
    rv = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_condition[i]:
            rv[arr_pos[i]] += 1
    return rv


@nb.njit
def base_contaminate_vertices(arr_new_infected, rand, level):
    counter = 0
    for i in range(arr_new_infected.shape[0]):
        if arr_new_infected[i]:
            if rand[counter] >= level:
                arr_new_infected[i] = False
            counter += 1

# ----------------------------------------------------------------------------------------------------------------------
# transition


@nb.njit
def transition_initialize_counters_of_newly_infected(arr_new_infected, arr_cnt, arr_new_cnt):
    counter = 0
    for i in range(arr_new_infected.shape[0]):
        if arr_new_infected[i]:
            arr_cnt[i] = arr_new_cnt[counter]
            counter += 1


@nb.njit
def transition_conditional_count_nb_agent_per_vertex(arr_condition, arr_pos, nb_vertex):
    rv = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_condition[i]:
            rv[arr_pos[i]] += 1
    return rv


@nb.njit
def transition_falsify_when_condition(arr_bool, condition):
    for i in range(arr_bool.shape[0]):
        if condition[i]:
            arr_bool[i] = False


# ----------------------------------------------------------------------------------------------------------------------
# transmission


@nb.njit
def transmission_contact_contagion_contact_tracing(arr_id, arr_pos, arr_con, rand_con, arr_new_infected, nb_vertex):
    nb_new_infected = arr_new_infected.sum()

    list_vert_id_con = List()
    for i in range(nb_vertex):
        con_on_pos_i = List()
        con_on_pos_i.append(arr_id[0])
        con_on_pos_i.pop()
        list_vert_id_con.append(con_on_pos_i)

    arr_nb_con_per_vertex = np.full((nb_vertex,), 0, dtype=np.int32)
    for i in range(arr_id.shape[0]):
        if arr_con[i]:
            arr_nb_con_per_vertex[arr_pos[i]] += 1
            list_vert_id_con[arr_pos[i]].append(arr_id[i])

    rv_id = np.full((nb_new_infected,), 0, dtype=np.int32)
    rv_con = np.full((nb_new_infected,), 0, dtype=np.int32)
    counter = 0
    for i in range(arr_new_infected.shape[0]):
        if arr_new_infected[i]:
            rand_index_con = int(np.floor(rand_con[counter] * arr_nb_con_per_vertex[arr_pos[i]])) % \
                             arr_nb_con_per_vertex[arr_pos[i]]
            rv_id[counter] = arr_id[i]
            rv_con[counter] = list_vert_id_con[arr_pos[i]][rand_index_con]
            counter += 1

    return rv_id, rv_con


