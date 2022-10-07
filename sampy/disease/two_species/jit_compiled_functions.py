import numpy as np
import numba as nb


@nb.njit
def base_contaminate_vertices(arr_new_infected, rand, level):
    counter = 0
    for i in range(arr_new_infected.shape[0]):
        if arr_new_infected[i]:
            if rand[counter] >= level:
                arr_new_infected[i] = False
            counter += 1


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


@nb.njit
def transmission_conditional_count(nb_vertex, arr_pos, arr_cond):
    pos_count = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_cond[i]:
            pos_count[arr_pos[i]] += 1
    return pos_count


@nb.njit
def transmission_count_needed_samples(susceptible, pos_susceptible, arr_count_con_host1, arr_count_con_host2):
    needed_samples = 0
    for i in range(susceptible.shape[0]):
        if susceptible[i]:
            needed_samples += (int(arr_count_con_host1[pos_susceptible[i]] > 0) +
                               int(arr_count_con_host2[pos_susceptible[i]] > 0))
    return needed_samples


@nb.njit
def transmission_disease_propagation(susceptible_host1, susceptible_host2, pos_host1, pos_host2, inf_status_host1,
                                     inf_status_host2, count_con_host1, count_con_host2, random_numbers,
                                     contact_rate_matrix):
    counter = 0
    for i in range(susceptible_host1.shape[0]):
        if susceptible_host1[i]:
            # host1 to host 1
            if count_con_host1[pos_host1[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[0][0]) ** count_con_host1[pos_host1[i]]:
                    inf_status_host1[i] = True
                counter += 1

            # host2 to host1
            if count_con_host2[pos_host1[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[1][0]) ** count_con_host2[pos_host1[i]]:
                    inf_status_host1[i] = True
                counter += 1

    for i in range(susceptible_host2.shape[0]):
        if susceptible_host2[i]:
            # host1 to host2
            if count_con_host1[pos_host2[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[0][1]) ** count_con_host1[pos_host2[i]]:
                    inf_status_host2[i] = True
                counter += 1

            # host2 to host2
            if count_con_host2[pos_host2[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[1][1]) ** count_con_host2[pos_host2[i]]:
                    inf_status_host2[i] = True
                counter += 1


@nb.njit
def transmission_disease_propagation_return_new_inf(susceptible_host1, susceptible_host2, pos_host1, pos_host2,
                                                    inf_status_host1, inf_status_host2, count_con_host1,
                                                    count_con_host2, random_numbers, contact_rate_matrix):
    counter = 0
    new_infected_host1 = np.full(susceptible_host1.shape, False)
    new_infected_host2 = np.full(susceptible_host2.shape, False)
    for i in range(susceptible_host1.shape[0]):
        if susceptible_host1[i]:
            # host1 to host 1
            if count_con_host1[pos_host1[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[0][0]) ** count_con_host1[pos_host1[i]]:
                    inf_status_host1[i] = True
                    new_infected_host1[i] = True
                counter += 1

            # host2 to host1
            if count_con_host2[pos_host1[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[1][0]) ** count_con_host2[pos_host1[i]]:
                    inf_status_host1[i] = True
                    new_infected_host1[i] = True
                counter += 1

    for i in range(susceptible_host2.shape[0]):
        if susceptible_host2[i]:
            # host1 to host2
            if count_con_host1[pos_host2[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[0][1]) ** count_con_host1[pos_host2[i]]:
                    inf_status_host2[i] = True
                    new_infected_host2[i] = True
                counter += 1

            # host2 to host2
            if count_con_host2[pos_host2[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[1][1]) ** count_con_host2[pos_host2[i]]:
                    inf_status_host2[i] = True
                    new_infected_host2[i] = True
                counter += 1

    return new_infected_host1, new_infected_host2


@nb.njit
def transmission_disease_propagation_return_type_inf(susceptible_host1, susceptible_host2, pos_host1, pos_host2,
                                                     inf_status_host1, inf_status_host2, count_con_host1,
                                                     count_con_host2, random_numbers, contact_rate_matrix):
    counter = 0
    type_inf_host1 = np.full(susceptible_host1.shape, 0, dtype=int)
    type_inf_host2 = np.full(susceptible_host2.shape, 0, dtype=int)
    for i in range(susceptible_host1.shape[0]):
        if susceptible_host1[i]:
            # host1 to host 1
            if count_con_host1[pos_host1[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[0][0]) ** count_con_host1[pos_host1[i]]:
                    inf_status_host1[i] = True
                    type_inf_host1[i] += 1
                counter += 1

            # host2 to host1
            if count_con_host2[pos_host1[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[1][0]) ** count_con_host2[pos_host1[i]]:
                    inf_status_host1[i] = True
                    type_inf_host1[i] += 2
                counter += 1

    for i in range(susceptible_host2.shape[0]):
        if susceptible_host2[i]:
            # host1 to host2
            if count_con_host1[pos_host2[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[0][1]) ** count_con_host1[pos_host2[i]]:
                    inf_status_host2[i] = True
                    type_inf_host2[i] += 1
                counter += 1

            # host2 to host2
            if count_con_host2[pos_host2[i]] > 0:
                if random_numbers[counter] < 1 - (1 - contact_rate_matrix[1][1]) ** count_con_host2[pos_host2[i]]:
                    inf_status_host2[i] = True
                    type_inf_host2[i] += 2
                counter += 1

    return type_inf_host1, type_inf_host2
