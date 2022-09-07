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
