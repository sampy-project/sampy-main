import numba as nb
import numpy as np


@nb.njit
def vaccination_apply_vaccine_from_array_condition(arr_vaccine_status, arr_cnt_vaccine_status, arr_immune_status,
                                                   arr_vaccine_level, arr_position, rand, condition):
    counter_rand = 0
    newly_vaccinated = np.full(arr_vaccine_status.shape, False, dtype=np.bool_)
    for i in range(arr_vaccine_status.shape[0]):
        if condition[i]:
            if rand[counter_rand] < arr_vaccine_level[arr_position[i]]:
                arr_vaccine_status[i] = True
                arr_cnt_vaccine_status[i] = 0
                arr_immune_status[i] = True
                newly_vaccinated[i] = True
            counter_rand += 1
    return newly_vaccinated


@nb.njit
def culling_apply_culling_from_array_condition(arr_culling_level, arr_position, rand, condition):
    counter_rand = 0
    survive = np.full(arr_position.shape, True, dtype=np.bool_)
    for i in range(arr_position.shape[0]):
        if condition[i]:
            if rand[counter_rand] < arr_culling_level[arr_position[i]]:
                survive[i] = False
            counter_rand += 1
    return survive
