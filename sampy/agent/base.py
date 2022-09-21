import pandas as pd
import numpy as np
from .jit_compiled_functions import (count_nb_agent_per_vertex,
                                     conditional_count_nb_agent_per_vertex)
from ..pandas_xs.pandas_xs import DataFrameXS


class BaseAgingAgent:
    """
    Base class for aging agents, i.e. agents having an age attribute that is used in the simulation (for instance for
    varying probabilities of dispersion or varying mortality rates).
    """
    def __init__(self, **kwargs):
        try:
            self.graph = kwargs['graph']
        except KeyError:
            raise ValueError("A graph object should be passed to the constructor, using the kwarg 'graph'.")

        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

        self.df_population['col_id'] = None
        self.df_population['age'] = None

        if hasattr(self, 'dict_default_val'):
            self.dict_default_val['age'] = 0
        else:
            self.dict_default_val = {'age': 0}

        self.counter_id = 0

        self.on_ticker = ['increase_age']
        self.type = 'agent'

    def _sampy_debug_add_attribute(self, name_attr, def_value=np.nan):
        if not isinstance(name_attr, str):
            raise ValueError("An attribute name should be a string.")
        if name_attr in self.df_population.dict_colname_to_index:
            raise KeyError("An attribute with the name " + name_attr + " already exists.")
        x = np.array([def_value])
        if x.shape != (1,):
            raise ValueError("A default value should be a single number, integer or np.nan.")
        if str(x.dtype) not in self.df_population.ALLOWED_TYPE:
            raise ValueError("The chosen default value results in arrays of dtype '" + str(x.dtype) + "', which is not"
                             " supported by DataFrameXS object. Supported types are:" +
                             str(self.df_population.LIST_ALLOWED_TYPE) + ".")

    def add_attribute(self, name_attr, def_value=np.nan):
        """
        add a new column to the dataframe df_population, whose name is the parameter 'name_attr'

        :param name_attr: name of the new column
        :param def_value: default value for the created column, set the whole column to this value if df_population is
            non empty, and save the default value for later calls.
        """
        self.df_population[name_attr] = def_value
        self.dict_default_val[name_attr] = def_value

    def _sampy_debug_set_default_val(self, dict_values, replace=False):
        if not hasattr(dict_values, 'items') or not hasattr(getattr(dict_values, 'items'), '__call__'):
            raise TypeError("Dict_value parameter should be a dictionary like object, with a method 'items' "
                             "allowing to loop over keys and values of the object.")
        for key, val in dict_values.items():
            if not isinstance(key, str):
                raise KeyError("Column names should be a string.")
            if key not in self.df_population.dict_colname_to_index:
                raise KeyError("Trying to set the default value of a non existing column (" + key + ")")
            x = np.array([val])
            if x.shape != (1,):
                raise ValueError("A default value should be a single number, integer or np.nan.")
            if str(x.dtype) not in self.df_population.ALLOWED_TYPE:
                raise ValueError(
                    "The chosen default value results in arrays of dtype '" + str(x.dtype) + "', which is not" +
                    " supported by DataFrameXS object. Supported types are:" +
                    str(self.df_population.LIST_ALLOWED_TYPE) + ".")

    def set_default_val(self, dict_values, replace=False):
        """
        Change the defaults values in the attribute dict_default_val. If replace is True, then the attr dict_default_val
        is replaced by the content of the argument dict_values. Otherwise, the content of dict_values is added to
        dict_values, modifying it if needed.

        :param dict_values: dictionary like object, with an items method
        :param replace: optional, boolean, default False
        """
        if replace:
            self.dict_default_val = {}
        for name, val in dict_values.items():
            self.dict_default_val[name] = val

    def _sampy_debug_add_agents(self, dict_values):
        if not hasattr(dict_values, 'items') or not hasattr(getattr(dict_values, 'items'), '__call__'):
            raise TypeError("Dict_value parameter should be a dictionary like object, with a method 'items' "
                             "allowing to loop over keys and values of the object.")
        nb_rows_added = 0
        found_non_cst_col = False
        name_first_non_cst_column = None
        for key, val in dict_values.items():
            if not isinstance(key, str):
                raise KeyError("Column names should be a string.")
            if key not in self.df_population.dict_colname_to_index:
                raise KeyError("The agents have an that is not a column of the population DataFrame (" + key + ").")
            x = np.array(val)
            if len(x.shape) > 1:
                raise ValueError("The value for column " + key + " results in an array of dimension " +
                                 str(x.shape) + " while add_agents expects one dimensional arrays or constants.")
            if len(x.shape) == 1:
                if found_non_cst_col:
                    if x.shape[0] != nb_rows_added:
                        raise ValueError("The value for column " + key + " results in creating " + str(x.shape[0]) +
                                         " lines, while column " + name_first_non_cst_column + " value results in " +
                                         "creating " + str(nb_rows_added) + '.')
                else:
                    nb_rows_added = x.shape[0]
                    found_non_cst_col = True
                    name_first_non_cst_column = key

    def add_agents(self, dict_values):
        """
        add new rows to the dataframe df_population, which corresponds to new individuals. The recommanded use for this
        method is to provide a dict_value of the form:

            dict_value = {name_col1: list_of_values1, name_col2: list_of_values2, ...}

        list of values can be replaced by np.array. Note that some of those lists can be replaced by constants if you
        want some columns to be filled with a single value. If the user provides constant for all list_of_values, then a
        single line is added. Finally, any non mentioned column will be filled with the associated default value.

        Note that the user does not have the hand on the col_id column since it is considered internal, and the user
        should add an extra column to identify some individuals.

        :param dict_values: values of the attributes of the new individuals.
        """

        constant_cols = {}
        array_cols = {}
        for name, col in dict_values.items():
            if col is None:
                array_cols[name] = col
            else:
                arr_col = np.array(col)
                if len(arr_col.shape) == 0:
                    constant_cols[name] = col
                else:
                    array_cols[name] = arr_col

        # create a DataFrameXS from the input values
        df = DataFrameXS()

        # populate the dataframe df. First considering the case where some lists have been provided.
        if array_cols:
            for name, col in array_cols.items():
                df[name] = col
            # then add the constant columns
            for name, const in constant_cols.items():
                df[name] = const

        # now the case where only single values have been provided
        else:
            for name, const in dict_values.items():
                if const is None or (len(np.array(const).shape) > 0 and np.array(const).shape[0] == 0) :
                    df[name] = None
                else:
                    df[name] = [const]

        # add the default values if needed.
        for name, value in self.dict_default_val.items():
            if name not in df.dict_colname_to_index:
                df[name] = value

        # create the col_id column
        df['col_id'] = [self.counter_id + i for i in range(df.nb_rows)]
        self.counter_id = df['col_id'][-1] + 1

        # concatenate df to df_population
        self.df_population.concat(df, inplace=True)

    def increase_age(self):
        """
        increment by one the age of all the agents
        """
        self.df_population['age'] += 1

    def tick(self):
        """
        execute in order all the methods whose name are in the list 'on_ticker'. Those methods should not accept
        any arguments.
        """
        for method in self.on_ticker:
            getattr(self, method)()

    def save_population_to_csv(self, path, sep=';', **kwargs):
        """
        Save the dataframe df_population as csv

        :param path: full path of the csv file
        :param sep: optional. Separator used in csv. Default is ';'
        """
        self.df_population.to_csv(path, sep=sep, **kwargs)

    def load_population_from_csv(self, path, sep=';', **kwargs):
        """
        load a csv film to create the dataframe. Override the existing dataframe df_population, if any.

        :param path: full path of the csv file
        :param sep: optional. Separator used in csv. Default is ';'
        """
        self.df_population = DataFrameXS.read_csv(path, sep=sep, **kwargs)

    def _sampy_debug_count_pop_per_vertex(self, position_attribute='position', condition=None):
        if condition is not None:
            if (not isinstance(condition, np.ndarray)) or \
                    (condition.shape != (self.df_population.nb_rows,)) or \
                    (not str(condition.dtype).startswith('bool')):
                raise ValueError("if used, condition argument should be a 1D array of bool of same length as the number"
                                 " of individuals.")

    def count_pop_per_vertex(self, position_attribute='position', condition=None):
        """
        Count the number of agent in each cell, and return the result as a 1D numpy array.

        :param position_attribute: Optional, string, default 'position'. Name of the attribute corresponding to cell
                                   index on which agent is.
        :param condition: Optional, 1D array of bool, default None. If not None, count only the agent for which the
                          condition is True.

        :return: 1D array X such that X[i] is the number of agent in the cell whose index is i.
        """
        if condition is None:
            return count_nb_agent_per_vertex(self.df_population[position_attribute],
                                             self.graph.weights.shape[0])
        else:
            if (not isinstance(condition, np.ndarray)) or \
               (len(condition.shape) != 1) or \
               (condition.shape[0] != self.df_population.nb_rows) or \
               (not str(condition.dtype).startswith('bool')):
                raise ValueError("if used, condition argument should be a 1D array of bool of same length as the number"
                                 " of individuals.")
            return conditional_count_nb_agent_per_vertex(condition,
                                                         self.df_population[position_attribute],
                                                         self.graph.weights.shape[0])

    @property
    def number_agents(self):
        return self.df_population.nb_rows
