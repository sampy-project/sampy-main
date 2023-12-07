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