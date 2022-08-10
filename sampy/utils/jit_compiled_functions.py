import numpy as np
import numba as nb


# General jited functions
@nb.jit(nopython=True)
def rand_sample_varying_weights(weights, target, rand):
    rv = np.empty((weights.shape[0],), dtype=np.int32)
    for i in range(weights.shape[0]):
        for j in range(weights[i].shape[0]):
            if rand[i] <= weights[i][j]:
                rv[i] = target[i][j]
                break
    return rv


# mortality related
@nb.jit
def update_mortality_orm_methodo(nb_vert, col_pos, col_K, col_age, col_proba_death):
    counter_pos = np.zeros((nb_vert,), dtype=np.int32)
    for i in range(col_pos.shape[0]):
        counter_pos[col_pos[i]] += 1
    rv = np.empty((col_pos.shape[0],), dtype=np.float64)
    for i in range(col_pos.shape[0]):
        rv[i] = col_proba_death[col_age[i]]*(counter_pos[col_pos[i]] / max(col_K[col_pos[i]], 0.01))
    return rv


@nb.njit
def update_mortality_sex_orm_method(nb_vert, col_pos, col_K, col_age, col_male, col_prob_death_male,
                                    col_prob_death_female):
    counter_pos = np.zeros((nb_vert,), dtype=np.int32)
    for i in range(col_pos.shape[0]):
        counter_pos[col_pos[i]] += 1
    rv = np.empty((col_pos.shape[0],), dtype=np.float64)
    for i in range(col_pos.shape[0]):
        if col_male[i]:
            rv[i] = col_prob_death_male[col_age[i]]*(counter_pos[col_pos[i]] / col_K[col_pos[i]])
        else:
            rv[i] = col_prob_death_female[col_age[i]] * (counter_pos[col_pos[i]] / col_K[col_pos[i]])
    return rv


# reproduction of the agent
@nb.njit
def reprod_part1(nb_vertex, position, gender):
    count_male = np.zeros((nb_vertex,), dtype=np.int32)
    count_female = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(position.shape[0]):
        if gender[i]:
            count_male[position[i]] += 1
        else:
            count_female[position[i]] += 1
    return count_male.max(), count_female.max()


@nb.njit
def reprod_part2(nb_vertex, max_male, max_female, arr_id, position, gender, rand):
    males = np.full((nb_vertex, max_male), -1, dtype=np.int32)
    females = np.full((nb_vertex, max_female), -1, dtype=np.int32)
    counter_males = np.zeros((nb_vertex,), dtype=np.int32)
    counter_females = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(rand.shape[0]):
        v = rand[i]
        u = position[v]
        if gender[v]:
            males[u][counter_males[u]] = v
            counter_males[u] += 1
        else:
            females[u][counter_females[u]] = v
            counter_females[u] += 1

    rv_bool = np.full((rand.shape[0],), False)
    rv_mate = np.full((rand.shape[0],), -1, dtype=np.int32)
    n = min(males.shape[1], females.shape[1])
    for i in range(nb_vertex):
        for j in range(n):
            if males[i][j] == -1 or females[i][j] == -1:
                break
            else:
                rv_bool[females[i][j]] = True
                rv_mate[females[i][j]] = arr_id[males[i][j]]
    return rv_bool, rv_mate


# movement of the agents
@nb.njit
def conditional_movement(territory, bool_if_mov, connections, weights, rand):
    rv = np.empty((territory.shape[0],), dtype=np.int32)
    for i in range(territory.shape[0]):
        if bool_if_mov[i]:
            neighb_not_found = True
            for j in range(weights.shape[1]):
                if rand[i] <= weights[territory[i]][j]:
                    rv[i] = connections[territory[i]][j]
                    neighb_not_found = False
                    break
                if neighb_not_found:
                    rv[i] = territory[i]
        else:
            rv[i] = territory[i]
    return rv


@nb.njit
def mandatory_movement(territory, connections, weights, rand):
    rv = np.empty((territory.shape[0],), dtype=np.int32)
    for i in range(territory.shape[0]):
        neighb_not_found = True
        for j in range(weights.shape[1]):
            if rand[i] <= weights[territory[i]][j]:
                rv[i] = connections[territory[i]][j]
                neighb_not_found = False
                break
        if neighb_not_found:
            rv[i] = territory[i]
    return rv


@nb.njit
def create_image_builtin_graph(first_dim, last_dim, arr_pos):
    r_img = np.full((first_dim, last_dim), 0, dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        r_img[arr_pos[i] // first_dim][arr_pos[i] % last_dim] += 1
    return r_img


@nb.njit
def create_image_from_count_array_builtin_graph(first_dim, last_dim, arr_count):
    r_img = np.full((first_dim, last_dim), 0, dtype=np.int32)
    if first_dim >= last_dim:
        for i in range(arr_count.shape[0]):
            r_img[i // first_dim, i % last_dim] = arr_count[i]
    else:
        for i in range(arr_count.shape[0]):
            r_img[i % first_dim, i // last_dim] = arr_count[i]
    return r_img


# disease related
@nb.njit
def fill_poisson_counter(arr_cnt, arr_new_infected, poisson_sample):
    r_cnt = np.zeros(arr_cnt.shape, dtype=np.int32)
    index_poisson_sample = 0
    for i in range(arr_cnt.shape[0]):
        if arr_new_infected[i]:
            r_cnt[i] = poisson_sample[index_poisson_sample]
            index_poisson_sample += 1
        else:
            r_cnt[i] = arr_cnt[i]
    return r_cnt


@nb.njit
def fill_custom_transition_counter(arr_cnt, arr_new_infected, arr_nb_timestep, arr_cum_prob, arr_uniform_sample):
    r_cnt = np.zeros(arr_cnt.shape, dtype=np.int32)
    index_uniform_sample = 0
    for i in range(arr_cnt.shape[0]):
        if arr_new_infected[i]:
            for j in range(arr_nb_timestep.shape[0]):
                if arr_uniform_sample[index_uniform_sample] <= arr_cum_prob[j]:
                    r_cnt[i] = arr_nb_timestep[j]
                    index_uniform_sample += 1
                    break
        else:
            r_cnt[i] = arr_cnt[i]
    return r_cnt


@nb.njit
def conditional_count_return_full_array_two_species(nb_vertex, arr_pos_source, arr_con_source, arr_pos_target):
    pos_count = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos_source.shape[0]):
        if arr_con_source[i]:
            pos_count[arr_pos_source[i]] += 1
    rv = np.zeros((arr_pos_target.shape[0],), dtype=np.int32)
    for i in range(arr_pos_target.shape[0]):
        rv[i] = pos_count[arr_pos_target[i]]
    return rv


# generic jit compiled functions
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


@nb.njit
def falsify_when_condition(arr_bool, arr_condition):
    for i in range(arr_bool.shape[0]):
        if arr_condition[i]:
            arr_bool[i] = False
    return arr_bool


@nb.njit
def zero_when_condition(arr_int, arr_condition):
    for i in range(arr_int.shape[0]):
        if arr_condition[i]:
            arr_int[i] = 0
    return arr_int


@nb.njit
def count_nb_agent_per_vertex(arr_pos, nb_vertex):
    rv = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        rv[arr_pos[i]] += 1
    return rv


@nb.njit
def conditional_count_nb_agent_per_vertex(arr_condition, arr_pos, nb_vertex):
    rv = np.zeros((nb_vertex,), dtype=np.int32)
    for i in range(arr_pos.shape[0]):
        if arr_condition[i]:
            rv[arr_pos[i]] += 1
    return rv
