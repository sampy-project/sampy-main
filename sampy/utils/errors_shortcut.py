import numpy as np


def check_input_array(array, name_argument, prefix_dtype, nb_dim=None, shape=None):
    if not isinstance(array, np.ndarray):
        raise TypeError("The parameter " + name_argument + " should be an array.")
    if not str(array.dtype).startswith(prefix_dtype):
        raise TypeError("The parameter " + name_argument + " should be an array of type " + prefix_dtype + ".")
    if nb_dim is not None:
        if len(array.shape) != nb_dim:
            raise ValueError("The parameter " + name_argument + " should be an array of dim " + str(nb_dim) +
                             ", while the input is of dim " + str(len(array.shape)) + ".")
    if shape is not None:
        if array.shape != shape:
            raise ValueError("The parameter " + name_argument + " should be an array of shape " + str(shape) +
                             ", while the input is of shape " + str(array.shape) + ".")


def check_input_is_permutation(array, name_argument, length):
    if (not isinstance(array, np.ndarray)) or (not str(array.dtype).startswith('int')):
        raise TypeError("The parameter " + name_argument + " should be an array of integers.")
    if array.shape != (length,):
        raise ValueError("The parameter " + name_argument + " should be an array of shape (nb_agents,).")
    if not (np.sort(array) == np.arange(0, length).all()):
        raise ValueError("The parameter " + name_argument + " should either be None, or a permutation of all "
                         "the integers from 0 to " + str(length - 1) + ".")


def check_col_exists_good_type(df, name_col, name_argument, prefix_dtype='', reject_none=False):
    if not isinstance(name_col, str):
        raise TypeError("A column name should be a string, which is not the case for provided " + name_argument + ".")
    if name_col not in df.dict_colname_to_index:
        raise KeyError("Provided " + name_argument + " does not match any column name in df_population.")
    if reject_none:
        if df[name_col] is None:
            raise ValueError("The column " + name_col + " is empty while it should contain the age of the agents.")
        if not str(df[name_col].dtype).startswith(prefix_dtype):
            raise TypeError("The column " + name_col + " is not of the proper type. Expected " + prefix_dtype +
                            ", got " + str(df[name_col].dtype) + ".")


def check_if_gender_array(array):
    arr_1 = array == 1
    arr_0 = array == 0
    if not (arr_1 | arr_0).all():
        raise TypeError("The provided column for gender should only contains 0s and 1s, since it is the way gender "
                        "is encoded in sampy.")
