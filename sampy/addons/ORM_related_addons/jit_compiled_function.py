import numpy as np
import numba as nb


@nb.njit
def orm_like_agent_orm_dispersion(arr_can_move, arr_will_move, arr_rand_nb_steps, arr_rand_direction,
                                  col_position, col_territory, col_has_moved, connections):
    counter_arr_will_move = 0
    counter_arr_rand_nb_steps = 0
    counter_arr_rand_direction = 0
    for i in range(arr_can_move.shape[0]):
        if arr_can_move[i]:

            # first we determine if the agent is allowed to move. If not, we skip to the next agent
            will_move = arr_will_move[counter_arr_will_move]
            counter_arr_will_move += 1
            if not will_move:
                continue

            # now we update the 'has moved' flag (in ORM, this is done even if the agent moves 0 cells)
            col_has_moved[i] = True

            nb_steps = arr_rand_nb_steps[counter_arr_rand_nb_steps]
            counter_arr_rand_nb_steps += 1
            if nb_steps == 0:
                continue

            for j in range(nb_steps):
                rand_direction = arr_rand_direction[counter_arr_rand_direction]
                counter_arr_rand_direction += 1

                if j == 0:
                    direction = int(np.floor(rand_direction * 6)) % 6
                else:
                    if rand_direction <= 0.2:
                        direction = (direction - 1) % 6
                    elif rand_direction >= 0.8:
                        direction = (direction + 1) % 6

                if connections[col_territory[i]][direction] == -1:
                    break

                col_position[i] = connections[col_territory[i]][direction]
                col_territory[i] = connections[col_territory[i]][direction]


@nb.njit
def orm_like_agent_dispersion_with_reflexion(arr_can_move, arr_will_move, arr_rand_nb_steps, arr_rand_direction,
                                             col_position, col_territory, col_has_moved, connections, weights):
    counter_arr_will_move = 0
    counter_arr_rand_nb_steps = 0
    counter_arr_rand_direction = 0
    for i in range(arr_can_move.shape[0]):
        if arr_can_move[i]:

            # first we determine if the agent is allowed to move. If not, we skip to the next agent
            will_move = arr_will_move[counter_arr_will_move]
            counter_arr_will_move += 1
            if not will_move:
                continue

            # now we update the 'has moved' flag (in ORM, this is done even if the agent moves 0 cells)
            col_has_moved[i] = True

            nb_steps = arr_rand_nb_steps[counter_arr_rand_nb_steps]
            counter_arr_rand_nb_steps += 1
            if nb_steps == 0:
                continue

            for j in range(nb_steps):
                rand_direction = arr_rand_direction[counter_arr_rand_direction]
                counter_arr_rand_direction += 1

                if j == 0:
                    direction = int(np.floor(rand_direction * 6)) % 6
                else:
                    if rand_direction <= 0.2:
                        direction = (direction - 1) % 6
                    elif rand_direction >= 0.8:
                        direction = (direction + 1) % 6

                if connections[col_territory[i]][direction] == -1:
                    for k in range(6):
                        if rand_direction <= weights[col_territory[i]][k]:
                            col_position[i] = connections[col_territory[i]][k]
                            col_territory[i] = connections[col_territory[i]][k]
                            break
                else:
                    col_position[i] = connections[col_territory[i]][direction]
                    col_territory[i] = connections[col_territory[i]][direction]


@nb.njit
def orm_like_agent_orm_dispersion_with_resistance(arr_can_move, arr_will_move, arr_rand_nb_steps, arr_rand_direction,
                                                  col_position, col_territory, col_has_moved, connections,
                                                  prob_successful_mov, rand_res):
    counter_arr_will_move = 0
    counter_arr_rand_nb_steps = 0
    counter_arr_rand_direction = 0
    counter_rand_res = -1
    for i in range(arr_can_move.shape[0]):
        if arr_can_move[i]:

            # first we determine if the agent is allowed to move. If not, we skip to the next agent
            will_move = arr_will_move[counter_arr_will_move]
            counter_arr_will_move += 1
            if not will_move:
                continue
            # now we update the 'has moved' flag (in ORM, this is done even if the agent moves 0 cells)
            col_has_moved[i] = True

            nb_steps = arr_rand_nb_steps[counter_arr_rand_nb_steps]
            counter_arr_rand_nb_steps += 1
            if nb_steps == 0:
                continue

            for j in range(nb_steps):

                rand_direction = arr_rand_direction[counter_arr_rand_direction]
                counter_arr_rand_direction += 1

                if j == 0:
                    direction = int(np.floor(rand_direction * 6)) % 6
                else:
                    if rand_direction <= 0.2:
                        direction = (direction - 1) % 6
                    elif rand_direction >= 0.8:
                        direction = (direction + 1) % 6

                if connections[col_territory[i]][direction] == -1:
                    break

                counter_rand_res += 1
                if rand_res[counter_rand_res] >= prob_successful_mov[col_territory[i]][direction]:
                    break

                col_position[i] = connections[col_territory[i]][direction]
                col_territory[i] = connections[col_territory[i]][direction]


@nb.njit
def orm_like_agents_mov_around_with_resistance(territory, position, connections, weights, prob_successful_mov,
                                               arr_bool_mov, rand_direction, rand_res):
    counter = 0
    for i in range(territory.shape[0]):
        if arr_bool_mov[i]:
            for j in range(6):
                if rand_direction[counter] <= weights[territory[i]][j]:
                    if rand_res[counter] <= prob_successful_mov[territory[i]][j]:
                        position[i] = connections[territory[i]][j]
                    counter += 1
                    break


@nb.njit
def orm_like_agents_mortality_from_v08_no_condition_no_alpha_beta(arr_count, arr_condition_count, arr_position, arr_k,
                                                                  arr_annual_mortality, arr_age, arr_rand):
    rv = np.full(arr_age.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if arr_k[arr_position[i]] == 0.:
            rv[i] = False
            if arr_condition_count[i]:
                arr_count[arr_position[i]] -= 1
            continue

        mort_adjust = float(arr_count[arr_position[i]]) / float(arr_k[arr_position[i]])
        age_agent = arr_age[i] // 52
        p0 = 1 - np.exp(-arr_annual_mortality[age_agent])
        alpha = 4. * 1.5 / p0
        if arr_rand[i] <= (p0 / (1 + np.exp(-(mort_adjust - 1.5) * alpha))):
            rv[i] = False
            if arr_condition_count[i]:
                arr_count[arr_position[i]] -= 1
    return rv


@nb.njit
def orm_like_agents_mortality_from_v08_with_gender_no_condition_no_alpha_beta(arr_count, arr_condition_count,
                                                                              arr_position, arr_k, arr_gender,
                                                                              arr_female_mortality, arr_male_mortality,
                                                                              arr_age, arr_rand):
    rv = np.full(arr_age.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if arr_k[arr_position[i]] == 0.:
            rv[i] = False
            if arr_condition_count[i]:
                arr_count[arr_position[i]] -= 1
            continue

        mort_adjust = float(arr_count[arr_position[i]]) / float(arr_k[arr_position[i]])
        age_agent = arr_age[i] // 52
        if arr_gender[i] == 0:
            p0 = 1 - np.exp(-arr_male_mortality[age_agent])
        else:
            p0 = 1 - np.exp(-arr_female_mortality[age_agent])
        alpha = 4. * 1.5 / p0
        if arr_rand[i] <= (p0 / (1 + np.exp(-(mort_adjust - 1.5) * alpha))):
            rv[i] = False
            if arr_condition_count[i]:
                arr_count[arr_position[i]] -= 1
    return rv


def orm_like_agents_mortality_from_v08_with_gender_with_condition_no_alpha_beta(arr_count, arr_condition_count,
                                                                                arr_position, arr_k, arr_gender,
                                                                                arr_female_mortality, arr_male_mortality,
                                                                                arr_age, arr_rand, condition):
    rv = np.full(arr_age.shape, True, dtype=np.bool_)
    counter = 0
    for i in range(rv.shape[0]):
        if condition[i]:
            if arr_k[arr_position[i]] == 0.:
                rv[i] = False
                if arr_condition_count[i]:
                    arr_count[arr_position[i]] -= 1
                counter += 1
                continue

            mort_adjust = float(arr_count[arr_position[i]]) / float(arr_k[arr_position[i]])
            age_agent = arr_age[i] // 52
            if arr_gender[i] == 0:
                p0 = 1 - np.exp(-arr_male_mortality[age_agent])
            else:
                p0 = 1 - np.exp(-arr_female_mortality[age_agent])
            alpha = 4. * 1.5 / p0
            if arr_rand[counter] <= (p0 / (1 + np.exp(-(mort_adjust - 1.5) * alpha))):
                rv[i] = False
                if arr_condition_count[i]:
                    arr_count[arr_position[i]] -= 1
            counter += 1
    return rv


@nb.njit
def orm_like_agents_mortality_from_v08_with_condition_no_alpha_beta(arr_count, arr_condition_count, arr_position, arr_k,
                                                                    arr_annual_mortality, arr_age, arr_rand,
                                                                    condition):
    rv = np.full(arr_age.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if condition[i]:
            if arr_k[arr_position[i]] == 0.:
                rv[i] = False
                if arr_condition_count[i]:
                    arr_count[arr_position[i]] -= 1
                continue

            mort_adjust = float(arr_count[arr_position[i]]) / float(arr_k[arr_position[i]])
            age_agent = arr_age[i] // 52
            p0 = 1 - np.exp(-arr_annual_mortality[age_agent])
            alpha = 4. * 1.5 / p0
            if arr_rand[i] <= (p0 / (1 + np.exp(-(mort_adjust - 1.5) * alpha))):
                rv[i] = False
                if arr_condition_count[i]:
                    arr_count[arr_position[i]] -= 1
    return rv


@nb.njit
def orm_mongooses_check_if_mother_free_of_juveniles(offsprings_mom_id, mother_id, potential_mother):
    set_offsprings_mom_id = set(offsprings_mom_id)
    for i in range(potential_mother.shape[0]):
        if potential_mother[i]:
            if mother_id[i] in set_offsprings_mom_id:
                potential_mother[i] = False


@nb.njit
def orm_mongooses_update_ter_pos_youngs(mom_id, youngs, col_territory, col_position, col_id, col_mom_id):
    set_mom_id = set(mom_id)
    dict_mom_to_territory = dict()
    for i in range(col_id.shape[0]):
        if col_id[i] in set_mom_id:
            dict_mom_to_territory[col_id[i]] = col_territory[i]
    for i in range(youngs.shape[0]):
        if youngs[i]:
            col_territory[i] = dict_mom_to_territory[col_mom_id[i]]
            col_position[i] = dict_mom_to_territory[col_mom_id[i]]


@nb.njit
def orm_mongooses_update_mating_week(current_week, mean_week, sd_week, arr_next_mating,
                                     arr_init_to_perform, rand_gauss):
    counter = 0
    for i in range(arr_init_to_perform.shape[0]):
        if arr_init_to_perform[i]:
            potential_weeks = np.sort(np.rint(rand_gauss[counter] * sd_week + mean_week).astype('int32') % 52)
            next_week_found = False
            for j in range(potential_weeks.shape[0]):
                if current_week < potential_weeks[j]:
                    next_week_found = True
                    arr_next_mating[i] = potential_weeks[j]
                    break
            if not next_week_found:
                arr_next_mating[i] = potential_weeks[0]
            counter += 1


@nb.njit
def experimental_density_mortality(arr_count, arr_position, arr_age, arr_k, arr_gender, beta_male, beta_female,
                                   alpha, rand):
    rv = np.full(arr_age.shape, True, dtype=np.bool_)
    for i in range(rv.shape[0]):
        if arr_k[arr_position[i]] == 0.:
            rv[i] = False
            arr_count[arr_position[i]] -= 1
            continue

        mort_adjust = float(arr_count[arr_position[i]]) / float(arr_k[arr_position[i]])
        if arr_gender[i] == 0:
            p = 1. / (1. + np.exp(-alpha * (mort_adjust - beta_male[arr_age[i]])))
        else:
            p = 1. / (1. + np.exp(-alpha * (mort_adjust - beta_female[arr_age[i]])))

        if rand[i] <= p:
            rv[i] = False
            arr_count[arr_position[i]] -= 1
    return rv

