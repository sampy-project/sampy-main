import numba as nb
import numpy as np


@nb.njit
def expand_array_according_to_condition(arr_to_expand, arr_condition, default_val):
    """
    This function takes a 1D array a of any values, and a 1D array of booleans. The resulting array
    is of shape arr_condition.shape, and contains the values of arr_to_expand.
    """
    rv = np.full(arr_condition.shape, default_val, dtype=arr_to_expand.dtype)
    counter = 0
    for i in range(arr_condition.shape[0]):
        if arr_condition[i]:
            rv[i] = arr_to_expand[counter]
            counter += 1
    return rv


@nb.njit
def allocate_homes_to_agents_basic(arr_random, arr_capacity, number_agents, arr_home_indexes, 
                                   arr_nb_occupants):
    """
    This function assign a home to each agent. The methodology is quite straightforward, but can 
    lead to rather unrealistic situation (for instance a home populated only by agents whose ages 
    are under 18). Therefore, this should be considered as a first draft mainly for testing 
    purposes, and other home allocation methods will be added later.

    note that the array of home indexes will be completely transformed.

    IMPORTANT: this function is JIT compiled since, on a regular utilization of the model we will
               want to generate many different populations in order to analyze statistically the 
               output. However, this will have a performance cost on single run.
    """
    rv = np.full((number_agents,), -1, dtype=int)

    nb_non_empty_house = arr_home_indexes.shape[0]
    for i in range(arr_random.shape[0]):
        index_home_in_arr = int(np.floor(nb_non_empty_house * arr_random[i]))
        index_home = arr_home_indexes[index_home_in_arr]
        rv[i] = index_home
        arr_nb_occupants[index_home] += 1
        
        # here we modify arr_home_indexes when the current house reached full capacity.
        # That is, we remove the index from the array, and we pad all the following values one step
        # to the left.
        if arr_nb_occupants[index_home] == arr_capacity[index_home]:
            nb_non_empty_house -= 1
            for j in range(index_home_in_arr, arr_home_indexes.shape[0] - 1):
                if arr_home_indexes[j + 1] == -1:
                    arr_home_indexes[j] = -1
                    break
                else:
                    arr_home_indexes[j] = arr_home_indexes[j + 1]

    return rv