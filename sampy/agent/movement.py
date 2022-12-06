import numpy as np
from .jit_compiled_functions import *
from ..pandas_xs.pandas_xs import DataFrameXS
from ..utils.errors_shortcut import (check_col_exists_good_type,
                                     check_input_array)


class TerritorialMovementWithoutResistance:
    """
    Add graph based movements abilities to the agents. Agents have both a territory and a position, which can be
    different. Generally, the position of an agent will be either its territory vertex, or a vertex neighbouring its
    territory.

    Note that both position and territory are stored as integers, i.e. using the vertices indices which generally
    differ from their id. If needed, conversion should be performed by the user using the graph attribute
    'dict_cell_id_to_ind'. Equivalently, the user may extract a table giving the correspondence between id and indexes
    using the graph method 'save_table_id_of_vertices_to_indices'.
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

        self.df_population['territory'] = None
        self.df_population['position'] = None

    def _sampy_debug_change_territory(self,
                                      condition=None,
                                      territory_attribute='territory',
                                      position_attribute='position'):
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            if (not isinstance(condition, np.ndarray)) or \
                    (condition.shape != (self.df_population.nb_rows,)) or \
                    (not str(condition.dtype).startswith('bool')):
                raise ValueError("if used, condition argument should be a 1D array of bool of same length as the number"
                                 " of individuals.")

        check_col_exists_good_type(self.df_population, territory_attribute, prefix_dtype='int')
        check_col_exists_good_type(self.df_population, position_attribute, prefix_dtype='int')

    def change_territory(self,
                         condition=None,
                         territory_attribute='territory',
                         position_attribute='position'):
        """
        Change the territory and the position of the agents. If an agent is on an isolated vertex (a vertex without
        any neighbour), then the agent stays on the vertex.

        :param condition: optional, array of bool, default None. If not None, array telling which
        :param territory_attribute: optional, string, default 'territory'
        :param position_attribute: optional, string, default 'position'
        """
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            rand = np.random.uniform(0, 1, (condition.sum(),))
            movement_change_territory_and_position_condition(self.df_population[territory_attribute],
                                                             self.df_population[position_attribute],
                                                             condition, rand,
                                                             self.graph.connections, self.graph.weights)
        else:
            rand = np.random.uniform(0, 1, (self.df_population.shape[0],))
            movement_change_territory_and_position(self.df_population[territory_attribute],
                                                   self.df_population[position_attribute],
                                                   rand, self.graph.connections, self.graph.weights)

    def _sampy_debug_mov_around_territory(self,
                                          proba_remain_on_territory,
                                          condition=None,
                                          territory_attribute='territory',
                                          position_attribute='position'):
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))
        check_col_exists_good_type(self.df_population, territory_attribute, 'territory_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', prefix_dtype='int',
                                   reject_none=True)

    def mov_around_territory(self,
                             proba_remain_on_territory,
                             condition=None,
                             territory_attribute='territory',
                             position_attribute='position'):
        """
        Update the average position of the agent around its territory during the current time step.

        :param proba_remain_on_territory: float, probability to stay on the territory
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move.
        :param territory_attribute: optional, string, default 'territory'
        :param position_attribute: optional, string, default 'position'
        """
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            pre_bool_mov = np.random.uniform(0, 1, condition.sum()) > proba_remain_on_territory
            bool_mov = movement_mov_around_territory_fill_bool_mov_using_condition(pre_bool_mov, condition)
        else:
            bool_mov = np.random.uniform(0, 1, self.df_population.shape[0]) > proba_remain_on_territory
        rand = np.random.uniform(0, 1, bool_mov.sum())
        movement_mov_around_territory(self.df_population[territory_attribute], self.df_population[position_attribute],
                                      bool_mov, rand, self.graph.connections, self.graph.weights)

    def _sampy_debug_dispersion_with_varying_nb_of_steps(self, arr_nb_steps, arr_prob,
                                                         condition=None,
                                                         territory_attribute='territory',
                                                         position_attribute='position'
                                                         ):
        if self.df_population.nb_rows == 0:
            return

        check_input_array(arr_nb_steps, 'arr_nb_steps', 'int', nb_dim=1)
        check_input_array(arr_prob, 'arr_prob', 'float', nb_dim=1)

        if arr_prob.shape != arr_nb_steps.shape:
            raise ValueError("Arguments 'arr_nb_steps' and 'arr_prob' have different shapes.")

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

        check_col_exists_good_type(self.df_population, territory_attribute, 'territory_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', prefix_dtype='int',
                                   reject_none=True)

    def dispersion_with_varying_nb_of_steps(self, arr_nb_steps, arr_prob,
                                            condition=None,
                                            territory_attribute='territory',
                                            position_attribute='position'
                                            ):
        """
        Used to modelize dispersion of agents. Each selected agent will perform a random number of discrete steps on
        the graph. The number of steps is determined using the user inputs 'arr_nb_steps' and 'arr_prob'. Both
        position and territory are updated.

        :param arr_nb_steps: 1D array of int, giving the permissible number of steps.
        :param arr_prob: 1D array of float, arr_prob[i] is the probability that a given agent will perform
                         arr_nb_steps[i] steps.
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move. If left at None, all the agents move.
        :param territory_attribute: optional, string, default 'territory'.
        :param position_attribute: optional, string, default 'position'.
        """
        if self.df_population.nb_rows == 0:
            return
        prob = arr_prob.astype('float64')
        prob = prob / prob.sum()
        if condition is not None:
            # get number of steps
            arr_nb_steps = np.random.choice(arr_nb_steps, condition.sum(), p=prob)
        else:
            arr_nb_steps = np.random.choice(arr_nb_steps, self.df_population.nb_rows, p=prob)
        rand = np.random.uniform(0, 1, arr_nb_steps.sum())
        if condition is None:
            movement_dispersion_with_varying_nb_of_steps(self.df_population[territory_attribute],
                                                         self.df_population[position_attribute],
                                                         rand, arr_nb_steps, self.graph.connections, self.graph.weights)
        else:
            movement_dispersion_with_varying_nb_of_steps_condition(self.df_population[territory_attribute],
                                                                   self.df_population[position_attribute],
                                                                   condition,
                                                                   rand, arr_nb_steps, self.graph.connections,
                                                                   self.graph.weights)


class TerritorialMovementWithResistance:
    """
    Add graph based movements abilities to the agents. Agents have both a territory and a position, which can be
    different. Generally, the position of an agent will be either its territory vertex, or a vertex neighbouring its
    territory. Here each connection in the graph is assumed to come with probability saying how likely a movement on
    that connection is to fail.

    Note that both position and territory are stored as integers, i.e. using the vertices indices which generally
    differ from their id. If needed, conversion should be performed by the user using the graph attribute
    'dict_cell_id_to_ind'. Equivalently, the user may extract a table giving the correspondence between id and indexes
    using the graph method 'save_table_id_of_vertices_to_indices'.
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

        self.df_population['territory'] = None
        self.df_population['position'] = None

    def _sampy_debug_change_territory_with_resistance(self,
                                                      resistance_array,
                                                      condition=None,
                                                      territory_attribute='territory',
                                                      position_attribute='position'):
        if self.df_population.nb_rows == 0:
            return
        if (not isinstance(resistance_array, np.ndarray) or \
                resistance_array.shape != self.graph.connections.shape or \
                (not str(resistance_array.dtype).startswith('float'))):
            raise ValueError("The resistance array should be a 2 dimensional array of floats of shape " +
                             str(resistance_array.shape))
        if condition is not None:
            if (not isinstance(condition, np.ndarray)) or \
                    (condition.shape != (self.df_population.nb_rows,)) or \
                    (not str(condition.dtype).startswith('bool')):
                raise ValueError("if used, condition argument should be a 1D array of bool of same length as the number"
                                 " of individuals.")

        check_col_exists_good_type(self.df_population, territory_attribute, prefix_dtype='int')
        check_col_exists_good_type(self.df_population, position_attribute, prefix_dtype='int')

    def change_territory_with_resistance(self,
                                         resistance_array,
                                         condition=None,
                                         territory_attribute='territory',
                                         position_attribute='position'):
        """
        Change the territory and the position of the agents. If an agent is on an isolated vertex (a vertex without
        any neighbour), then the agent stays on the vertex.

        :param resistance_array: 2d array of float, array of same shape as the connections of the graph, gives the
                                 'resistance to movement' of each connection
        :param condition: optional, array of bool, default None. If not None, array telling which
        :param territory_attribute: optional, string, default 'territory'
        :param position_attribute: optional, string, default 'position'
        """
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            rand = np.random.uniform(0, 1, (condition.sum(),))
            movement_change_territory_and_position_condition(self.df_population[territory_attribute],
                                                             self.df_population[position_attribute],
                                                             condition, rand,
                                                             self.graph.connections, self.graph.weights)
        else:
            rand = np.random.uniform(0, 1, (self.df_population.shape[0],))
            movement_change_territory_and_position(self.df_population[territory_attribute],
                                                   self.df_population[position_attribute],
                                                   rand, self.graph.connections, self.graph.weights)

    def _sampy_debug_mov_around_territory(self,
                                          proba_remain_on_territory,
                                          condition=None,
                                          territory_attribute='territory',
                                          position_attribute='position'):
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))
        check_col_exists_good_type(self.df_population, territory_attribute, 'territory_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', prefix_dtype='int',
                                   reject_none=True)

    def mov_around_territory(self,
                             proba_remain_on_territory,
                             condition=None,
                             territory_attribute='territory',
                             position_attribute='position'):
        """
        Update the average position of the agent around its territory during the current time step.

        :param proba_remain_on_territory: float, probability to stay on the territory
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move.
        :param territory_attribute: optional, string, default 'territory'
        :param position_attribute: optional, string, default 'position'
        """
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            pre_bool_mov = np.random.uniform(0, 1, condition.sum()) > proba_remain_on_territory
            bool_mov = movement_mov_around_territory_fill_bool_mov_using_condition(pre_bool_mov, condition)
        else:
            bool_mov = np.random.uniform(0, 1, self.df_population.shape[0]) > proba_remain_on_territory
        rand = np.random.uniform(0, 1, bool_mov.sum())
        movement_mov_around_territory(self.df_population[territory_attribute], self.df_population[position_attribute],
                                      bool_mov, rand, self.graph.connections, self.graph.weights)

    def _sampy_debug_dispersion_with_varying_nb_of_steps(self, arr_nb_steps, arr_prob,
                                                         condition=None,
                                                         territory_attribute='territory',
                                                         position_attribute='position'
                                                         ):
        if self.df_population.nb_rows == 0:
            return

        check_input_array(arr_nb_steps, 'arr_nb_steps', 'int', nb_dim=1)
        check_input_array(arr_prob, 'arr_prob', 'float', nb_dim=1)

        if arr_prob.shape != arr_nb_steps.shape:
            raise ValueError("Arguments 'arr_nb_steps' and 'arr_prob' have different shapes.")

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

        check_col_exists_good_type(self.df_population, territory_attribute, 'territory_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', prefix_dtype='int',
                                   reject_none=True)

    def dispersion_with_varying_nb_of_steps(self, arr_nb_steps, arr_prob,
                                            condition=None,
                                            territory_attribute='territory',
                                            position_attribute='position'
                                            ):
        """
        Used to modelize dispersion of agents. Each selected agent will perform a random number of discrete steps on
        the graph. The number of steps is determined using the user inputs 'arr_nb_steps' and 'arr_prob'. Both
        position and territory are updated.

        :param arr_nb_steps: 1D array of int, giving the permissible number of steps.
        :param arr_prob: 1D array of float, arr_prob[i] is the probability that a given agent will perform
                         arr_nb_steps[i] steps.
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move. If left at None, all the agents move.
        :param territory_attribute: optional, string, default 'territory'.
        :param position_attribute: optional, string, default 'position'.
        """
        if self.df_population.nb_rows == 0:
            return
        prob = arr_prob.astype('float64')
        prob = prob / prob.sum()
        if condition is not None:
            # get number of steps
            arr_nb_steps = np.random.choice(arr_nb_steps, condition.sum(), p=prob)
        else:
            arr_nb_steps = np.random.choice(arr_nb_steps, self.df_population.nb_rows, p=prob)
        rand = np.random.uniform(0, 1, arr_nb_steps.sum())
        if condition is None:
            movement_dispersion_with_varying_nb_of_steps(self.df_population[territory_attribute],
                                                         self.df_population[position_attribute],
                                                         rand, arr_nb_steps, self.graph.connections, self.graph.weights)
        else:
            movement_dispersion_with_varying_nb_of_steps_condition(self.df_population[territory_attribute],
                                                                   self.df_population[position_attribute],
                                                                   condition,
                                                                   rand, arr_nb_steps, self.graph.connections,
                                                                   self.graph.weights)
