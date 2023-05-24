import numba as nb
import numpy as np
from numba.typed import List


# ---------------------------------------------------------------------------------------------------------------------
# base section


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

# ---------------------------------------------------------------------------------------------------------------------
# mortality section


@nb.njit
def mortality_natural_death_orm_methodology(bias, rand, arr_prob_male, arr_prob_female, arr_count, pos, k, age, gender):
    rv = np.full(rand.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if k[pos[i]] == 0:
            p = 1.
        else:
            if gender[i] == 0:
                p = arr_prob_male[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
            else:
                p = arr_prob_female[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
        if rand[i] <= p:
            rv[i] = False
            arr_count[pos[i]] -= 1
    return rv


@nb.njit
def mortality_natural_death_orm_methodology_condition_count(bias, rand, arr_prob_male, arr_prob_female,
                                                            arr_count, arr_cond_count, pos, k, age, gender):
    rv = np.full(rand.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if k[pos[i]] == 0:
            p = 1.
        else:
            if gender[i] == 0:
                p = arr_prob_male[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
            else:
                p = arr_prob_female[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
        if rand[i] <= p:
            rv[i] = False
            if arr_cond_count[i]:
                arr_count[pos[i]] -= 1
    return rv


@nb.njit
def mortality_natural_death_orm_methodology_condition_death(bias, rand, arr_prob_male, arr_prob_female,
                                                            arr_count, arr_cond_death, pos, k, age, gender):
    rv = np.full(rand.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if arr_cond_death[i]:
            if k[pos[i]] == 0:
                p = 1.
            else:
                if gender[i] == 0:
                    p = arr_prob_male[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
                else:
                    p = arr_prob_female[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
            if rand[i] <= p:
                rv[i] = False
                arr_count[pos[i]] -= 1
    return rv


@nb.njit
def mortality_natural_death_orm_methodology_both_cond(bias, rand, arr_prob_male, arr_prob_female,
                                                      arr_count, arr_cond_death, arr_cond_count, pos, k, age, gender):
    rv = np.full(rand.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if arr_cond_death[i]:
            if k[pos[i]] == 0:
                p = 1.
            else:
                if gender[i] == 0:
                    p = arr_prob_male[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
                else:
                    p = arr_prob_female[age[i]] * ((float(arr_count[pos[i]]) / float(k[pos[i]])) + bias)
            if rand[i] <= p:
                rv[i] = False
                if arr_cond_count[i]:
                    arr_count[pos[i]] -= 1
    return rv

# ---------------------------------------------------------------------------------------------------------------------
# reproduction


@nb.njit
def reproduction_find_random_mate_on_position(col_mate, col_pregnancy, arr_id, position, gender, nb_vertex,
                                              rand_preg, prob_pregnancy):
    list_vert_id_male = List()
    list_vert_index_male = List()
    for i in range(nb_vertex):
        male_on_pos_i = List()
        male_on_pos_i.append(arr_id[0])
        male_on_pos_i.pop()
        list_vert_id_male.append(male_on_pos_i)

        index_male_on_pos_i = List()
        index_male_on_pos_i.append(0)
        index_male_on_pos_i.pop()
        list_vert_index_male.append(index_male_on_pos_i)

    arr_nb_male_per_vertex = np.full((nb_vertex,), 0, dtype=np.int32)
    for i in range(arr_id.shape[0]):
        col_mate[i] = -1
        col_pregnancy[i] = False
        if gender[i] == 0:
            arr_nb_male_per_vertex[position[i]] += 1
            list_vert_id_male[position[i]].append(arr_id[i])
            list_vert_index_male[position[i]].append(i)

    arr_ind_vertex = np.full((nb_vertex,), 0, dtype=np.int32)
    counter = 0
    for i in range(arr_id.shape[0]):
        if gender[i] == 1:
            if arr_ind_vertex[position[i]] < arr_nb_male_per_vertex[position[i]]:
                col_mate[i] = list_vert_id_male[position[i]][arr_ind_vertex[position[i]]]
                col_mate[list_vert_index_male[position[i]][arr_ind_vertex[position[i]]]] = arr_id[i]
                arr_ind_vertex[position[i]] += 1
                if rand_preg[counter] <= prob_pregnancy:
                    col_pregnancy[i] = True
            counter += 1


@nb.njit
def reproduction_find_random_mate_on_position_condition(col_mate, col_pregnancy, arr_id, position, gender, nb_vertex,
                                                        rand_preg, prob_pregnancy, condition):
    list_vert_id_male = List()
    list_vert_index_male = List()
    for i in range(nb_vertex):
        male_on_pos_i = List()
        male_on_pos_i.append(arr_id[0])
        male_on_pos_i.pop()
        list_vert_id_male.append(male_on_pos_i)

        index_male_on_pos_i = List()
        index_male_on_pos_i.append(0)
        index_male_on_pos_i.pop()
        list_vert_index_male.append(index_male_on_pos_i)

    arr_nb_male_per_vertex = np.full((nb_vertex,), 0, dtype=np.int32)
    for i in range(arr_id.shape[0]):
        if condition[i]:
            col_mate[i] = -1
            col_pregnancy[i] = False
            if gender[i] == 0:
                arr_nb_male_per_vertex[position[i]] += 1
                list_vert_id_male[position[i]].append(arr_id[i])
                list_vert_index_male[position[i]].append(i)

    arr_ind_vertex = np.full((nb_vertex,), 0, dtype=np.int32)
    counter = 0
    for i in range(arr_id.shape[0]):
        if gender[i] == 1 and condition[i]:
            if arr_ind_vertex[position[i]] < arr_nb_male_per_vertex[position[i]]:
                col_mate[i] = list_vert_id_male[position[i]][arr_ind_vertex[position[i]]]
                col_mate[list_vert_index_male[position[i]][arr_ind_vertex[position[i]]]] = arr_id[i]
                arr_ind_vertex[position[i]] += 1
                if rand_preg[counter] <= prob_pregnancy:
                    col_pregnancy[i] = True
            counter += 1


@nb.njit
def reproduction_find_random_mate_on_position_polygamous(arr_id, position, gender, col_mate, col_pregnant,
                                                         nb_vertex, rand_preg, rand_chose_mate, prob_pregnancy):
    list_vert_id_male = List()
    for i in range(nb_vertex):
        male_on_pos_i = List()
        male_on_pos_i.append(arr_id[0])
        male_on_pos_i.pop()
        list_vert_id_male.append(male_on_pos_i)

    arr_nb_male_per_vertex = np.full((nb_vertex,), 0, dtype=np.int32)
    for i in range(arr_id.shape[0]):
        col_pregnant[i] = False
        col_mate[i] = -1
        if gender[i] == 0:
            arr_nb_male_per_vertex[position[i]] += 1
            list_vert_id_male[position[i]].append(arr_id[i])

    counter = 0
    for i in range(arr_id.shape[0]):
        if gender[i] == 1:
            if arr_nb_male_per_vertex[position[i]] > 0:
                rand_index_male = int(np.floor(rand_chose_mate[counter] * arr_nb_male_per_vertex[position[i]])) % \
                                  arr_nb_male_per_vertex[position[i]]
                col_mate[i] = list_vert_id_male[position[i]][rand_index_male]
                if rand_preg[counter] <= prob_pregnancy:
                    col_pregnant[i] = True
            counter += 1


@nb.njit
def reproduction_find_random_mate_on_position_polygamous_condition(arr_id, position, gender, col_mate, col_pregnant,
                                                                   nb_vertex, rand_preg, rand_chose_mate,
                                                                   prob_pregnancy, condition):
    list_vert_id_male = List()
    for i in range(nb_vertex):
        male_on_pos_i = List()
        male_on_pos_i.append(arr_id[0])
        male_on_pos_i.pop()
        list_vert_id_male.append(male_on_pos_i)

    arr_nb_male_per_vertex = np.full((nb_vertex,), 0, dtype=np.int32)
    for i in range(arr_id.shape[0]):
        if condition[i]:
            col_pregnant[i] = False
            col_mate[i] = -1
            if gender[i] == 0:
                arr_nb_male_per_vertex[position[i]] += 1
                list_vert_id_male[position[i]].append(arr_id[i])

    counter = 0
    for i in range(arr_id.shape[0]):
        if gender[i] == 1 and condition[i]:
            if arr_nb_male_per_vertex[position[i]] > 0:
                rand_index_male = int(np.floor(rand_chose_mate[counter] * arr_nb_male_per_vertex[position[i]])) % \
                                  arr_nb_male_per_vertex[position[i]]
                col_mate[i] = list_vert_id_male[position[i]][rand_index_male]
                if rand_preg[counter] <= prob_pregnancy:
                    col_pregnant[i] = True
            counter += 1

# ---------------------------------------------------------------------------------------------------------------------
# graph based movement section


@nb.njit
def movement_change_territory_and_position_condition(territory, position, condition, rand, connections, weights):
    counter_rand = 0
    for i in range(territory.shape[0]):
        if condition[i]:
            found = False
            for j in range(weights.shape[1]):
                if rand[counter_rand] <= weights[territory[i]][j]:
                    found = True
                    # very important to update position first
                    position[i] = connections[territory[i]][j]
                    territory[i] = connections[territory[i]][j]
                    break
            if not found:
                position[i] = territory[i]
            counter_rand += 1


@nb.njit
def movement_change_territory_and_position(territory, position, rand, connections, weights):
    for i in range(territory.shape[0]):
        found = False
        for j in range(weights.shape[1]):
            if rand[i] <= weights[territory[i]][j]:
                found = True
                position[i] = connections[territory[i]][j]
                territory[i] = connections[territory[i]][j]
                break
        if not found:
            position[i] = territory[i]


@nb.njit
def movement_mov_around_territory_fill_bool_mov_using_condition(pre_bool_mov, condition):
    counter = 0
    bool_mov = np.full(condition.shape, False, dtype=np.bool_)
    for i in range(condition.shape[0]):
        if condition[i]:
            if pre_bool_mov[counter]:
                bool_mov[i] = True
            counter += 1
    return bool_mov


@nb.njit
def movement_mov_around_territory(territory, position, bool_mov, rand, connections, weights):
    counter_rand = 0
    for i in range(territory.shape[0]):
        if bool_mov[i]:
            for j in range(weights.shape[1]):
                if rand[counter_rand] <= weights[territory[i]][j] and connections[territory[i]][j] != -1:
                    position[i] = connections[territory[i]][j]
                    break
            counter_rand += 1


@nb.njit
def movement_dispersion_with_varying_nb_of_steps_condition(territory, position, condition, rand, arr_nb_steps,
                                                           connections, weights):
    counter_rand = 0
    counter_arr_steps = 0
    for i in range(territory.shape[0]):
        if condition[i]:
            position[i] = territory[i]
            for _ in range(arr_nb_steps[counter_arr_steps]):
                for j in range(weights.shape[1]):
                    if rand[counter_rand] <= weights[territory[i]][j]:
                        position[i] = connections[territory[i]][j]
                        territory[i] = connections[territory[i]][j]
                        break
                counter_rand += 1
            counter_arr_steps += 1


@nb.njit
def movement_dispersion_with_varying_nb_of_steps(territory, position, rand, arr_nb_steps,
                                                 connections, weights):
    counter_rand = 0
    counter_arr_steps = 0
    for i in range(territory.shape[0]):
        position[i] = territory[i]
        for _ in range(arr_nb_steps[counter_arr_steps]):
            for j in range(weights.shape[1]):
                if rand[counter_rand] <= weights[territory[i]][j]:
                    position[i] = connections[territory[i]][j]
                    territory[i] = connections[territory[i]][j]
                    break
            counter_rand += 1
        counter_arr_steps += 1

# ---------------------------------------------------------------------------------------------------------------------
# spherical random walk section


@nb.njit
def random_walk_on_sphere_set_position_based_on_graph(arr_selected_agents, arr_pos_agent,
                                                      agent_coord_x, agent_coord_y, agent_coord_z,
                                                      graph_coord_x, graph_coord_y, graph_coord_z):
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            ind_vertex = arr_pos_agent[i]
            agent_coord_x[i] = graph_coord_x[ind_vertex]
            agent_coord_y[i] = graph_coord_y[ind_vertex]
            agent_coord_z[i] = graph_coord_z[ind_vertex]


@nb.njit
def random_walk_on_sphere_start_random_walk_uniform_prob(arr_start_rw, arr_iorw):
    for i in range(arr_start_rw.shape[0]):
        if arr_start_rw[i]:
            arr_iorw[i] = True


@nb.njit
def random_walk_on_sphere_start_random_walk_uniform_prob_return_new_walkers(arr_start_rw, arr_iorw):
    arr_new_walkers = np.full(arr_iorw.shape, False, dtype=np.bool_)
    for i in range(arr_start_rw.shape[0]):
        if arr_start_rw[i]:
            if not arr_iorw[i]:
                arr_new_walkers[i] = True
            arr_iorw[i] = True
    return arr_new_walkers


@nb.njit
def conditional_random_walk_on_sphere_start_random_walk_uniform_prob(arr_start_rw, arr_iorw, condition):
    counter_arr_start_rw = 0
    for i in range(arr_start_rw.shape[0]):
        if condition[i]:
            if arr_start_rw[counter_arr_start_rw]:
                arr_iorw[i] = True
            counter_arr_start_rw += 1


@nb.njit
def conditional_random_walk_on_sphere_start_random_walk_uniform_prob_return_new_walkers(arr_start_rw, arr_iorw,
                                                                                        condition):
    arr_new_walkers = np.full(arr_iorw.shape, False, dtype=np.bool_)
    counter_arr_start_rw = 0
    for i in range(arr_start_rw.shape[0]):
        if condition[i]:
            if arr_start_rw[counter_arr_start_rw]:
                if not arr_iorw[i]:
                    arr_new_walkers[i] = True
                arr_iorw[i] = True
            counter_arr_start_rw += 1
    return arr_new_walkers


@nb.njit
def random_walk_on_sphere_set_initial_dir_to_north(arr_selected_agents,
                                                   pos_x, pos_y, pos_z,
                                                   dir_x, dir_y, dir_z):
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            if pos_x[i] == 0. and pos_y[i] == 0:
                dir_x[i] = 1.
                dir_y[i] = 0.
                dir_z[i] = 0.
            else:
                norm_p = np.sqrt(pos_x[i]**2 + pos_y[i]**2 + pos_z[i]**2)
                position = np.array([pos_x[i], pos_y[i], pos_z[i]])
                position = position / norm_p
                # the direction toward the north at position is given, up to normalization, by the following formula
                # (0, 0, 1) - <(0, 0, 1), position> position
                # which leads to the following line
                direction = np.array([-position[2] * position[0],
                                      -position[2] * position[1],
                                      1. - position[2] ** 2])
                dir_norm = np.sqrt((direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2))
                dir_x[i] = direction[0] / dir_norm
                dir_y[i] = direction[1] / dir_norm
                dir_z[i] = direction[2] / dir_norm


@nb.njit
def random_walk_on_sphere_deviate_direction_from_angles(deviation_angle, arr_selected_agents, px, py, pz,
                                                        dx, dy, dz):
    counter_dev_angle = 0
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            angle = deviation_angle[counter_dev_angle]
            norm_pos = np.sqrt(px[i]**2 + py[i]**2 + pz[i]**2)
            position = np.array([px[i], py[i], pz[i]])
            position = position / norm_pos

            # we rotate the direction by an angle of Theta with respect to the rotation axis given by the position.
            # position and direction are assumed to be orthonormal. We use the Euler-Rodrigues Formula.
            c = np.cos(angle)
            s = np.sin(angle)
            direction = np.array([c * dx[i] + s * (position[1] * dz[i] - position[2] * dy[i]),
                                  c * dy[i] + s * (position[2] * dx[i] - position[0] * dz[i]),
                                  c * dz[i] + s * (position[0] * dy[i] - position[1] * dx[i])])

            # normalizing the result to avoid accumulation of approximation errors.
            direction = direction - np.dot(position, direction) * position
            norm_d = np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)
            direction = direction / norm_d

            # saving the results
            dx[i] = direction[0]
            dy[i] = direction[1]
            dz[i] = direction[2]

            # increment the counter
            counter_dev_angle += 1


@nb.njit
def random_walk_on_sphere_propose_step_gamma_law(arr_selected_agents, gamma_sample, pos_x, pos_y, pos_z, dir_x, dir_y, dir_z,
                                       radius):
    r_pos_x = np.full(gamma_sample.shape, -1.)
    r_pos_y = np.full(gamma_sample.shape, -1.)
    r_pos_z = np.full(gamma_sample.shape, -1.)

    r_dir_x = np.full(gamma_sample.shape, -1.)
    r_dir_y = np.full(gamma_sample.shape, -1.)
    r_dir_z = np.full(gamma_sample.shape, -1.)

    counter = 0
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            angle = gamma_sample[counter] / radius
            nmz_pos = np.array([pos_x[i], pos_y[i], pos_z[i]]) / np.sqrt(pos_x[i]**2 + pos_y[i]**2 + pos_z[i]**2)
            direction = np.array([dir_x[i], dir_y[i], dir_z[i]])

            nmz_new_pos = np.cos(angle) * nmz_pos + np.sin(angle) * direction
            nmz_new_pos = nmz_new_pos / np.sqrt(nmz_new_pos[0] ** 2 + nmz_new_pos[1]**2 + nmz_new_pos[2]**2)

            direction = np.cos(angle) * direction - np.sin(angle) * nmz_pos
            # this next line is just there to be absolutely sure we stay normal to the position
            direction = direction - (direction[0] * nmz_new_pos[0] + direction[1] * nmz_new_pos[1] +
                                     direction[2] * nmz_new_pos[2]) * nmz_new_pos
            # extra normalization because this step will be repeated many time
            direction = direction / np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)

            r_pos_x[counter] = radius * nmz_new_pos[0]
            r_pos_y[counter] = radius * nmz_new_pos[1]
            r_pos_z[counter] = radius * nmz_new_pos[2]

            r_dir_x[counter] = direction[0]
            r_dir_y[counter] = direction[1]
            r_dir_z[counter] = direction[2]

            counter += 1

    return r_pos_x, r_pos_y, r_pos_z, r_dir_x, r_dir_y, r_dir_z


@nb.njit
def random_walk_on_sphere_make_step_gamma_law(arr_selected_agents, gamma_sample, pos_x, pos_y, pos_z, dir_x, dir_y,
                                              dir_z, radius):
    counter = 0
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            angle = gamma_sample[counter] / radius
            nmz_pos = np.array([pos_x[i], pos_y[i], pos_z[i]]) / np.sqrt(pos_x[i]**2 + pos_y[i]**2 + pos_z[i]**2)
            direction = np.array([dir_x[i], dir_y[i], dir_z[i]])

            nmz_new_pos = np.cos(angle) * nmz_pos + np.sin(angle) * direction
            nmz_new_pos = nmz_new_pos / np.sqrt(nmz_new_pos[0] ** 2 + nmz_new_pos[1]**2 + nmz_new_pos[2]**2)

            direction = np.cos(angle) * direction - np.sin(angle) * nmz_pos
            # this next line is just there to be absolutely sure we stay normal to the position
            direction = direction - (direction[0] * nmz_new_pos[0] + direction[1] * nmz_new_pos[1] +
                                     direction[2] * nmz_new_pos[2]) * nmz_new_pos
            # extra normalization because this step will be repeated many time
            direction = direction / np.sqrt(direction[0]**2 + direction[1]**2 + direction[2]**2)

            pos_x[i] = radius * nmz_new_pos[0]
            pos_y[i] = radius * nmz_new_pos[1]
            pos_z[i] = radius * nmz_new_pos[2]

            dir_x[i] = direction[0]
            dir_y[i] = direction[1]
            dir_z[i] = direction[2]

            counter += 1


@nb.njit
def random_walk_on_sphere_validate_step(arr_selected_agent, arr_success, proposed_px, proposed_py, proposed_pz,
                                        proposed_dx, proposed_dy, proposed_dz, px, py, pz, dx, dy, dz):
    counter = 0
    for i in range(arr_selected_agent.shape[0]):
        if arr_selected_agent[i]:
            if arr_success[counter]:
                px[i] = proposed_px[counter]
                py[i] = proposed_py[counter]
                pz[i] = proposed_pz[counter]
                dx[i] = proposed_dx[counter]
                dy[i] = proposed_dy[counter]
                dz[i] = proposed_dz[counter]
            counter += 1


@nb.njit
def random_walk_on_sphere_validate_step_return_fail(arr_selected_agent, arr_success, proposed_px, proposed_py,
                                                    proposed_pz, proposed_dx, proposed_dy, proposed_dz, px, py, pz,
                                                    dx, dy, dz):
    returned_arr = np.full(arr_selected_agent.shape, False)
    counter = 0
    for i in range(arr_selected_agent.shape[0]):
        if arr_selected_agent[i]:
            if arr_success[counter]:
                px[i] = proposed_px[counter]
                py[i] = proposed_py[counter]
                pz[i] = proposed_pz[counter]
                dx[i] = proposed_dx[counter]
                dy[i] = proposed_dy[counter]
                dz[i] = proposed_dz[counter]
            else:
                arr_selected_agent[i] = True
            counter += 1
    return returned_arr


@nb.njit
def _temp_random_walk_on_sphere_exit_random_walk_based_on_k(arr_selected_agents, rand, prob, alpha, arr_pos, arr_k,
                                                            arr_pop):
    rv = np.full(arr_selected_agents.shape, False, dtype=np.bool_)
    counter = 0
    for i in range(arr_selected_agents.shape[0]):
        if arr_selected_agents[i]:
            if rand[counter] <= prob * np.exp(-alpha * (arr_pop[arr_pos[i]] / arr_k[arr_pos[i]])):
                rv[i] = True
            counter += 1
    return rv
