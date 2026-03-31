import numba as nb
import numpy as np
from numba.typed import List

# ---------------------------------------------------------------------------------------------------------------------
# graph based movement section


@nb.njit
def movement_directional_dispersion_with_varying_nb_of_steps(territory, position, arr_nb_steps, arr_cumul_directional_prob, 
                                                             connections, weights, arr_rand_step, arr_rand_res, col_direction, 
                                                             arr_in_res, arr_out_res, nb_retry_res, stop_mov):
    counter_rand_step = 0 
    counter_rand_res = 0
    for i in range(arr_nb_steps.shape[0]): 
        for j in range(arr_nb_steps[i]): 
            succes_step = False
            for _ in range(nb_retry_res + 1):
                if col_direction[i] == -1: 
                    for k in range(weights.shape[1]): 
                        if arr_rand_step[counter_rand_step] < weights[territory[i]][k]:
                            candidate_position = connections[territory[i]][k]
                            if arr_rand_res[counter_rand_res] < (1-arr_out_res[territory[i]])*(1-arr_in_res[candidate_position]): 
                                position[i] = connections[territory[i]][k]
                                territory[i] = connections[territory[i]][k]
                                col_direction[i] = k
                                succes_step = True
                            break
                    counter_rand_res += 1        
                    counter_rand_step += 1
                else: 
                    for k in range(arr_cumul_directional_prob.shape[0]): 
                        if arr_rand_step[counter_rand_step] < arr_cumul_directional_prob[k]:
                            counter_rand_step += 1
                            candidate_position = connections[territory[i]][(col_direction[i]-3 + k) % 6]
                            if candidate_position == -1: 
                                for l in range(weights.shape[1]):
                                    if arr_rand_step[counter_rand_step] < weights[territory[i]][l]:
                                        new_candidate_position = connections[territory[i]][l]
                                        if arr_rand_res[counter_rand_res] < (1-arr_out_res[territory[i]])*(1-arr_in_res[new_candidate_position]):  
                                            position[i] = connections[territory[i]][l]
                                            territory[i] = connections[territory[i]][l]
                                            col_direction[i] = l
                                            succes_step = True
                                        break
                                counter_rand_res += 1
                                counter_rand_step += 1
                            else:
                                if arr_rand_res[counter_rand_res] < (1-arr_out_res[territory[i]])*(1-arr_in_res[candidate_position]): 
                                    position[i] = candidate_position
                                    territory[i] = candidate_position
                                    col_direction[i] = (col_direction[i]-3 + k) % 6
                                    succes_step = True
                                else:
                                    col_direction[i] = -1
                                counter_rand_res += 1
                            break
                if succes_step:
                    break
            if stop_mov and (not succes_step): 
                break


@nb.njit
def movement_directional_dispersion_with_varying_nb_of_steps_return_path(territory, position, arr_nb_steps, arr_cumul_directional_prob, 
                                                                         connections, weights, arr_rand_step, arr_rand_res, 
                                                                         col_direction, col_id, update_dir, arr_in_res, arr_out_res, 
                                                                         nb_retry_res, stop_mov):
    counter_rand_step = 0
    counter_rand_res = 0
    r_dict = dict()
    for i in range(arr_nb_steps.shape[0]):
        if arr_nb_steps[i] > 0:
            r_dict[col_id[i]] = np.full(arr_nb_steps[i] + 1, -1, dtype=connections.dtype)
            r_dict[col_id[i]][0] = territory[i]
        for j in range(arr_nb_steps[i]): 
            succes_step = False
            for _ in range(nb_retry_res + 1):
                if col_direction[i] == -1: 
                    for k in range(weights.shape[1]): 
                        if arr_rand_step[counter_rand_step] < weights[territory[i]][k]:
                            candidate_position = connections[territory[i]][k]
                            if arr_rand_res[counter_rand_res] < (1-arr_out_res[territory[i]])*(1-arr_in_res[candidate_position]):
                                # r_dict[col_id[i]][j + 1] = connections[territory[i]][k]
                                position[i] = connections[territory[i]][k]
                                territory[i] = connections[territory[i]][k]
                                col_direction[i] = k
                                succes_step = True
                            break
                    counter_rand_res += 1        
                    counter_rand_step += 1
                else: 
                    for k in range(arr_cumul_directional_prob.shape[0]): 
                        if arr_rand_step[counter_rand_step] < arr_cumul_directional_prob[k]:
                            counter_rand_step += 1
                            candidate_position = connections[territory[i]][(col_direction[i]-3 + k) % 6]
                            if candidate_position == -1: 
                                for l in range(weights.shape[1]):
                                    if arr_rand_step[counter_rand_step] < weights[territory[i]][l]:
                                        new_candidate_position = connections[territory[i]][l]
                                        if arr_rand_res[counter_rand_res] < (1-arr_out_res[territory[i]])*(1-arr_in_res[new_candidate_position]): 
                                            # r_dict[col_id[i]][j + 1] = connections[territory[i]][l] 
                                            position[i] = connections[territory[i]][l]
                                            territory[i] = connections[territory[i]][l]
                                            col_direction[i] = l
                                            succes_step = True
                                        break
                                counter_rand_res += 1
                                counter_rand_step += 1
                            else:
                                if arr_rand_res[counter_rand_res] < (1-arr_out_res[territory[i]])*(1-arr_in_res[candidate_position]): 
                                    # r_dict[col_id[i]][j + 1] = candidate_position
                                    position[i] = candidate_position
                                    territory[i] = candidate_position
                                    if update_dir: 
                                        col_direction[i] = (col_direction[i]-3 + k) % 6
                                    succes_step = True
                                else:
                                    col_direction[i] = -1
                                counter_rand_res += 1
                            break
 
                r_dict[col_id[i]][j + 1] = territory[i]
 
                if succes_step:
                    break

            if stop_mov and (not succes_step): 
                break

    return r_dict

#----------------------------------------------------
@nb.njit
def update_id_if_needed(col_id, col_border, dict_path, next_available_id, col_prob_change_id, arr_rand):
    new_id = next_available_id
    arr_has_change_id = np.full(col_id.shape, False) 
    arr_last_border_cell = np.full(col_id.shape, -1) 

    for i in range(col_id.shape[0]): 
        if col_id[i] not in dict_path: 
            continue

        actual_prob = 1.
        last_border_cell = -1
        for j in range(dict_path[col_id[i]].shape[0]): 
            if col_border[dict_path[col_id[i]][j]]:
                last_border_cell = dict_path[col_id[i]][j]
                actual_prob = actual_prob * (1 - col_prob_change_id[last_border_cell])
        
        actual_prob = 1. - actual_prob

        if arr_rand[i] < actual_prob:
            col_id[i] = new_id
            new_id += 1
            arr_has_change_id[i] = True
            arr_last_border_cell[i] = last_border_cell

    return new_id, arr_has_change_id, arr_last_border_cell

#----------------------------------------------------
@nb.njit
def update_rabies_if_needed_with_vacc(arr_has_changed_id, arr_last_border_cell, col_prob_change_rabies, 
                                      col_inf_status, col_inf_cnt, col_con_status, col_con_cnt, col_imm,
                                      arr_rand_has_rabies, 
                                      arr_nb_timestep_inf, 
                                      col_vacc_status, col_vac_cnt):
    counter_has_rabies = 0
    for i in range(arr_has_changed_id.shape[0]):
        if arr_has_changed_id[i]:
            # reset vac
            col_vacc_status[i] = False
            col_vac_cnt[i] = 0

            # reset imm
            col_imm[i] = False

            # reset inf
            col_inf_status[i] = False
            col_inf_cnt[i] = 0

            # reset con
            col_con_status[i] = False
            col_con_cnt[i] = 0

            # test if the new agents is infected
            if arr_rand_has_rabies[counter_has_rabies] < col_prob_change_rabies[arr_last_border_cell[i]]: # the agent has rabies
                col_inf_status[i] = True
                # now we need to pick the number of weeks the agent remains infected
                col_inf_cnt[i] = arr_nb_timestep_inf[counter_has_rabies]

            counter_has_rabies += 1

#----------------------------------------------------
@nb.njit
def update_rabies_if_needed(arr_has_changed_id, arr_last_border_cell, col_prob_change_rabies, 
                                      col_inf_status, col_inf_cnt, col_con_status, col_con_cnt, col_imm,
                                      arr_rand_has_rabies, 
                                      arr_nb_timestep_inf):
    counter_has_rabies = 0
    for i in range(arr_has_changed_id.shape[0]):
        if arr_has_changed_id[i]:
            # reset imm
            col_imm[i] = False

            # reset inf
            col_inf_status[i] = False
            col_inf_cnt[i] = 0

            # reset con
            col_con_status[i] = False
            col_con_cnt[i] = 0

            # test if the new agents is infected
            if arr_rand_has_rabies[counter_has_rabies] < col_prob_change_rabies[arr_last_border_cell[i]]: 
                col_inf_status[i] = True
                # now we need to pick the number of weeks the agent remains infected
                col_inf_cnt[i] = arr_nb_timestep_inf[counter_has_rabies]

            counter_has_rabies += 1