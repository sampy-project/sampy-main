import numpy as np
import copy
from .jit_compiled_functions import dataframe_xs_check_arr_in_col, dataframe_xs_check_arr_in_col_conditional
import pandas as pd


class DataFrameXS:
    """
    Class that encodes a really simplified version of a dataframe, where each column is stored as a separated one
    dimensional numpy array. Note that this class is made to be more rigid than Pandas Dataframe: the class won't try
    to fill in blanks or fancy things like that. The idea being to maintain control on what is going on and avoid bugs
    that goes unnoticed.

    Indexing:
    ---------
    3 types of indexing are supported: indexing by a string, an array of bool or an array of int. Any other index will
    raise a TypeError.
        - Indexing by a string returns the corresponding column. Keep in mind that it is not a copy, and modifying it
          will change the original column.
        - Indexing by an array of bool returns a copy.
        - Indexing by an array of int returns a copy.
    """
    LIST_ALLOWED_TYPE = ['bool', 'uint8', 'int8', 'uint16', 'int16', 'uint32', 'int32',
                         'uint64', 'int64', 'float32', 'float64']
    ALLOWED_TYPE = set(LIST_ALLOWED_TYPE)

    def __init__(self):
        self.list_col = []
        self.list_col_name = []
        self.dict_colname_to_index = dict()
        self.nb_rows = 0
        self.nb_cols = 0

# ----------------------------------------------------------------------------------------------------------------------
# class methods

    @classmethod
    def from_lists(cls, list_col_name, list_col):
        """
        Returns a DataFrameXS from a list of column names and a list of columns. If the lists are of different length,
        or if some column names are repeated, an Exception will be raised.

        :param list_col_name: list of strings
        :param list_col: list of 1D numpy array, that will become columns of the dataframe.

        :return: DataFrameXS object.
        """
        if len(list_col) != len(list_col_name):
            raise ValueError("Inputs do not have the same length.")
        if len(list_col_name) > len(set(list_col_name)):
            raise ValueError("Several columns have the same name.")

        r_df = cls()
        for name, col in zip(list_col_name, list_col):
            r_df[name] = col
        return r_df

    @classmethod
    def from_dict(cls, dict_name_to_col):
        """
        Returns a DataFrameXS object from a dictionary whose keys are column names, and values are columns.

        :param dict_name_to_col: dictionary, whose key are strings and values 1D numpy arrays or None.

        :return: DataFrameXS object
        """
        r_df = cls()
        for name, col in dict_name_to_col.items():
            r_df[name] = col
        return r_df

    @classmethod
    def read_csv(cls, path, sep=';', **kwargs):
        """
        Use pandas to read a csv, and then converts the pandas Dataframe into a DataFrameXS. Be careful that the type
        of the columns are supported by DataFrameXS instances.

        :param path: string, path to the csv
        :param sep: optional, string, default ';'. Separator in the csv.

        :return: DataFrameXS object
        """
        df = cls()
        pd_df = pd.read_csv(path, sep=sep, **kwargs)
        for name in pd_df:
            df[name] = np.copy(np.array(pd_df[name]))
        return df

# ----------------------------------------------------------------------------------------------------------------------
# __methods__

    def __setitem__(self, key, value):
        """
        Allow the user to set a column values by using the syntax df[key] = value

        :param key: string, name of the column
        :param value: either a single value, in which case all the column will be set to this value, or an iterable that
                      can be casted to a numpy array of supported type. If the Value is None, then the column
                      is set empty.
        """
        # check if the user provided a string as a column name
        if not isinstance(key, str):
            raise ValueError("A column name should be a string in a dataframe_xs.")

        # in case a column is set empty
        if value is None:
            self._add_empty_col(key)
        else:
            # cast the input as a numpy array
            temp_value = np.array(value)

            # check if the input is of allowed type
            if str(temp_value.dtype) not in self.ALLOWED_TYPE:
                raise ValueError("The proposed column is of an unsupported type. Supported types are " +
                                 str(self.LIST_ALLOWED_TYPE))

            # check if the user provided a single value
            if len(temp_value.shape) == 0:
                if self.nb_rows == 0:
                    self._add_empty_col(key)
                else:
                    value = np.full((self.nb_rows,), value)
                    self._add_col_without_nb_rows_check(key, value)

            # if the user provided an input that results in an array of shape (0,)
            elif temp_value.shape == (0,):
                self._add_empty_col(key)

            # Now the case of a real serie of values
            else:
                value = temp_value
                # check that the the input is of dimension 1
                if len(value.shape) > 1:
                    raise ValueError("A column should be an array of dimension 1, and your input dim is " +
                                     str(value.shape) + '.')

                # in the case the dataframe is empty
                if self.nb_rows == 0:
                    self.nb_rows = value.shape[0]
                    self._add_col_without_nb_rows_check(key, value)
                else:
                    if value.shape[0] != self.nb_rows:
                        raise ValueError("Dataframe has " + str(self.nb_rows) + " rows while the input has " +
                                         str(value.shape[0]) + '.')
                    else:
                        self._add_col_without_nb_rows_check(key, value)

    def __getitem__(self, item):
        """
        Use the syntax df[item] to retrieve data. Three possible uses:
            1) item is the name of a column: retrieve the corresponding column. Keep in mind that this is not a copy,
               and modifying the obtained array will modify the column in the dataframe.
            2) item is an array of bool of same length as the dataframe: retrieve a copy of the dataframe with only the
               lines of position i such that item[i] == True.
            3) item is an array of int: retrieve a copy of the dataframe

        :param item: either a string, a 1D array of bool, or a 1D array of int.

        :return: a column of the DataFrameXS if item is a string, a DataFrameXS object otherwise.
        """
        # if a string is provided, return the column as np.array
        if isinstance(item, str):
            if item not in self.dict_colname_to_index:
                raise KeyError("No column named " + item + ".")
            else:
                return self.list_col[self.dict_colname_to_index[item]]

        # if an array is provided
        elif isinstance(item, np.ndarray):
            # check if the array is empty
            if len(item.shape) == 0 or item.shape == (0,):
                raise IndexError("Cannot index using an empty array")

            # check if the array is not of dim 1
            if len(item.shape) != 1:
                raise IndexError("Boolean array indexing requires the array of boolean indexes to be one dimensional.")

            # case of an array of bool
            elif str(item.dtype) == 'bool':
                if item.shape[0] != self.nb_rows:
                    raise IndexError("Boolean array indexing requires the array of boolean indexes to be of the same" +
                                     "length as the dataframe.")
                nb_rows = item.sum()
                if nb_rows == 0:
                    r_df = DataFrameXS()
                    for name in self.list_col_name:
                        r_df[name] = None
                    return r_df
                else:
                    new_cols = []
                    for col in self.list_col:
                        if col is None:
                            new_cols.append(None)
                        else:
                            new_cols.append(col[item])
                    return self._create_df_without_check(self.list_col_name, new_cols, nb_rows, self.nb_cols)

            # case of an array of int
            elif str(item.dtype).startswith('int'):
                new_cols = []
                for name, col in zip(self.list_col_name, self.list_col):
                    if col is None:
                        new_cols.append(None)
                    else:
                        new_cols.append(col[item])
                return self._create_df_without_check(self.list_col_name, new_cols, item.shape[0], self.nb_cols)
            else:
                raise TypeError("Indexing with an array is only allowed using Bool or Int arrays.")
        else:
            raise TypeError("A Dataframe XS only supports integer, array of int and array of bool indexing.")

# ----------------------------------------------------------------------------------------------------------------------
# private methods

    def _add_empty_col(self, key):
        if key in self.dict_colname_to_index:
            self.list_col[self.dict_colname_to_index[key]] = None
            if self.is_empty:
                self.nb_rows = 0
        else:
            self.dict_colname_to_index[key] = self.nb_cols
            self.nb_cols += 1
            self.list_col_name.append(key)
            self.list_col.append(None)

    def _add_col_without_nb_rows_check(self, key, value):
        if key in self.dict_colname_to_index:
            self.list_col[self.dict_colname_to_index[key]] = value
        else:
            self.dict_colname_to_index[key] = self.nb_cols
            self.nb_cols += 1
            self.list_col_name.append(key)
            self.list_col.append(value)

    @classmethod
    def _create_df_without_check(cls, list_col_name, list_col, nb_rows, nb_cols):
        r_df = cls()
        r_df.list_col_name = copy.deepcopy(list_col_name)
        r_df.list_col = list_col
        r_df.nb_rows = nb_rows
        r_df.nb_cols = nb_cols
        r_df.dict_colname_to_index = {name: index for index, name in enumerate(list_col_name)}
        return r_df

# ----------------------------------------------------------------------------------------------------------------------
# properties

    @property
    def is_empty(self):
        for value in self.list_col:
            if value is not None:
                return False
        return True

    @property
    def shape(self):
        return self.nb_rows, self.nb_cols

# ----------------------------------------------------------------------------------------------------------------------
# methods

    def _sampy_debug_scramble(self, permutation=None, return_permutation=False):
        if permutation is not None:
            if not isinstance(permutation, np.ndarray):
                raise TypeError("Permutation parameter should be an array.")
            if not str(permutation.dtype).startswith('int'):
                raise TypeError("Permutation parameter should be an array of integers.")
            if not permutation.shape == (self.nb_rows,):
                raise ValueError("Permutation parameter should be an array of integers of shape (nb_rows,).")
            if not (np.sort(permutation) == np.arange(0, self.nb_rows)).all():
                raise ValueError("Permutation parameter should be a permutation of the integers from 0 to nb_rows.")

    def scramble(self, permutation=None, return_permutation=False):
        """
        Scramble the rows of the dataframe XS. The user can provide a permutation that will be used to scramble the
        rows.

        WARNING: if you use the keyword argument permutation, be sure you provide a permutation of the correct size.
        Otherwise the result can be unpredictable.

        :param permutation: optional, array of int, default None.
        :param return_permutation: optional, boolean default False. If True, the function returns the permutation used
                                   to shuffle the DataFrameXS (useful if some other objects have to be shuffled).
        """
        if permutation is None:
            permutation = np.random.permutation(self.nb_rows)
        for i, col in enumerate(self.list_col):
            if col is not None:
                self.list_col[i] = col[permutation]

        if return_permutation:
            return permutation

    def _sampy_debug_get_copy(self, col_name):
        if not isinstance(col_name, str):
            raise TypeError("get_copy expects a string as argument.")
        else:
            if col_name not in self.dict_colname_to_index:
                raise KeyError("no column with the name " + col_name + " in the dataframe.")

    def get_copy(self, col_name):
        """
        Returns a copy of the selected column.
        :param col_name: string, name of the column to copy.
        """
        col = self.list_col[self.dict_colname_to_index[col_name]]
        if col is None:
            return
        else:
            return np.copy(col)

    def _sampy_debug_check_arr_in_col(self, input_arr, name_col, condition=None):
        if condition is not None:
            if not isinstance(condition, np.ndarray):
                raise ValueError("condition should be an array.")
            if str(condition.dtype) != 'bool':
                raise ValueError("condition should be an array of bool.")
            if input_arr.shape != condition.shape:
                raise ValueError("condition and input_arr don't have the same shape.")

        if not isinstance(input_arr, np.ndarray):
            raise ValueError('Input array is not a numpy ndarray.')
        if len(input_arr.shape) != 1:
            raise ValueError('Input array is not one dimensional.')
        if input_arr.shape[0] == 0:
            raise ValueError('Input array is empty.')
        if str(input_arr.dtype) != str(self[name_col].dtype):
            raise ValueError("The input array (" + str(input_arr.dtype) + ") is not of the same type as the " +
                             "selected column (" + str(self[name_col].dtype) + ").")

    def check_arr_in_col(self, input_arr, name_col, condition=None):
        """
        Check which elements of the input array are in selected column.

        :param input_arr: 1D numpy array.
        :param name_col: string, name of the column
        :param condition: optional, boolean array.

        :return: 1D boolean array saying which elements of the input array are in the selected column.
        """
        if condition is None:
            return dataframe_xs_check_arr_in_col(input_arr, self[name_col])
        else:
            return dataframe_xs_check_arr_in_col_conditional(input_arr, self[name_col], condition)

    def _sampy_debug_change_type(self, col_name, str_type):
        if not isinstance(str_type, str):
            raise ValueError("str_type should be a string in the list: " + str(self.LIST_ALLOWED_TYPE) + ".")
        if str_type not in self.ALLOWED_TYPE:
            raise ValueError("The type " + str_type + "is not in " + str(self.LIST_ALLOWED_TYPE) + ".")

    def change_type(self, col_name, str_type):
        """
        Change the type of a column. The selected type has to be in the allowed type list.
        Do nothing if the selected column is empty.

        :param col_name: string, name of the column whose type should be changed.
        :param type: string, numpy identifier of the new type of the column.
        """
        if self[col_name] is not None:
            self[col_name] = self[col_name].astype(str_type)

    def copy(self):
        """
        Returns a copy of the dataframe
        :return: a dataframeXS
        """
        new_list_col = []
        for col in self.list_col:
            new_list_col.append(np.copy(col))
        return self._create_df_without_check(self.list_col_name, new_list_col, self.nb_rows, self.nb_cols)

    def concat(self, df, inplace=True):
        """
        Append the input DataFrameXS to the current DataFrameXS. The column names of the current DataFrameXS have to be
        column names of the input, or an Exception will be raised. Finally, note that the types of the columns of the
        input will be casted to the types of the corresponding columns of the current DataFrameXS.

        Extra columns in the input are ignored.

        :param df: DataFrame XS
        :param inplace: optional, boolean, default True. If True, the input is appended to the current DataFrame.
                        Otherwise, a new DataFrameXS is returned.

        :return: if inplace is False, return a DataFrameXS, otherwise returns None.
        """
        if not isinstance(df, DataFrameXS):
            raise ValueError("Input is not a DataframeXS")

        # case where the current DataFrameXS is empty, which can happen when initializing a simulation
        if self.is_empty:

            # if input is empty, the case is clear
            if df.is_empty:
                if inplace:
                    return
                else:
                    return self.copy()

            # if df is not empty, fill self columns with df columns
            else:
                new_col = []
                for name in self.list_col_name:
                    col = df[name]
                    if col is None:
                        new_col.append(None)
                    else:
                        new_col.append(np.copy(col))
                if inplace:
                    self.list_col = new_col
                    self.nb_rows = df.nb_rows
                else:
                    return self._create_df_without_check(self.list_col_name, new_col, df.nb_rows, self.nb_cols)

        # when both dataframes are not empty. Most important checks are done by __getitem__
        elif not df.is_empty:
            new_col = []
            for name in self.list_col_name:
                df_col = df[name]
                self_col = self[name]
                if self_col is not None:
                    if df_col is not None:
                        new_col.append(np.hstack([self_col, df_col]).astype(self_col.dtype))
                    else:
                        filled_col = np.full((df.nb_rows,), np.nan).astype(self_col.dtype)
                        new_col.append(np.hstack([self_col, filled_col]).astype(self_col.dtype))
                else:
                    if df_col is not None:
                        filled_col = np.full((self.nb_rows,), np.nan).astype(df_col.dtype)
                        new_col.append(np.hstack([filled_col, df_col]).astype(df_col.dtype))
                    else:
                        new_col.append(None)
            if inplace:
                self.nb_rows += df.nb_rows
                self.list_col = new_col
            else:
                return self._create_df_without_check(self.list_col_name, new_col, self.nb_rows + df.nb_rows,
                                                     self.nb_cols)

        # when current is not empty, and input is. No checks are done in this case.
        else:
            if inplace:
                return
            else:
                return self.copy()

    def add_rows_from_dict(self, dict_val, inplace=True):
        """
        Use "from_dict" class method to create a new DataFrameXS from the dict-like object in argument dict_val, and
        then concatenate it to the current DataFrameXS. If inplace is True, this methods returns nothing and modify the
        current DataFrameXS, otherwise it returns a new dataframe obtained by concatenation

        :param dict_val: dictionary like object, used to create a new df that will be concatenated to self.
        :param inplace: optional, boolean, default True. If true, current df is modified, otherwise returns a new df.

        :return: if inplace is True, returns None, else returns a DataFrameXS
        """
        # creates a DataFrame from the inputs
        df = DataFrameXS.from_dict(dict_val)

        # concat method takes care of modifying and returning what's needed.
        return self.concat(df, inplace=inplace)

    def get_as_pandas_dataframe(self):
        """
        Copy the dataframe content into a panda DataFrame, which is returned
        :return: Pandas DataFrame
        """
        pd_df = pd.DataFrame()
        for name in self.dict_colname_to_index:
            pd_df[name] = np.copy(self[name])
        return pd_df

    def to_csv(self, path, sep=';', **kwargs):
        """
        Save the DataFrame content into a CSV.

        WARNING: as a 'quick and dirty solution', this method calls 'get_as_pandas_dataframe' and then use pandas
        'to_csv' method. This is obviously inefficient, therefore this method should not be called regularly in a sim.
        """
        df = self.get_as_pandas_dataframe()
        df.to_csv(path, sep=sep, index=False, **kwargs)
