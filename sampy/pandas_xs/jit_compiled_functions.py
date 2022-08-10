import numba as nb
import numpy as np


@nb.njit
def dataframe_xs_check_arr_in_col(input_arr, column):
    rv = np.full(input_arr.shape, False, dtype=np.bool_)
    set_col = set(column)
    for i in range(input_arr.shape[0]):
        if input_arr[i] in set_col:
            rv[i] = True
    return rv


@nb.njit
def dataframe_xs_check_arr_in_col_conditional(input_arr, column, condition):
    rv = np.full(input_arr.shape, False, dtype=np.bool_)
    set_col = set(column)
    for i in range(input_arr.shape[0]):
        if condition[i]:
            if input_arr[i] in set_col:
                rv[i] = True
    return rv

