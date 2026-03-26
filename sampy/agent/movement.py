import numpy as np
from .jit_compiled_functions import *
from ..pandas_xs.pandas_xs import DataFrameXS
from ..utils.errors_shortcut import (check_col_exists_good_type,
                                     check_input_array)


class BasicMovement:
    """
    Add basic graph based movement abilities to the agents. Agents get a new position attribute, which stores
    a vertex id (i.e. an integer representing a unique vertex of the graph) corresponding to the vertex on
    which the agent live.
    """
    def __init__(self, **kwargs):
        # this check is in Theory not needed, since a 'BaseClass' should provide a df_population attribute
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

        self.df_population['position'] = None

    def change_position(self, condition=None, position_attribute='position'):
        """
        Change the position of the agents by making them move to a neighouring vertex. If an agent is on an
        isolated vertex (a vertex without any neighbour), then the agent stays on the vertex.

        :param condition: optional, array of bool, default None. If not None, array telling which agent can
                          change position.
        :param position_attribute: optional, string, default 'position'
        """
        if self.df_population.nb_rows == 0:
            return
        if condition is not None:
            rand = np.random.uniform(0, 1, (condition.sum(),))
            movement_change_position_condition(self.df_population[position_attribute], condition, rand,
                                               self.graph.connections, self.graph.weights)
        else:
            rand = np.random.uniform(0, 1, self.df_population.nb_rows)
            movement_change_position(self.df_population[position_attribute], rand,
                                     self.graph.connections, self.graph.weights)


class SwitchMovementCriterion:
    """
    Add methods that can be used to decided if an agent start or exit a movement.    
    """
    def __init__(self, **kwargs):
        pass

    def pick_agents_using_density_resources_logistic(self, D, beta, alpha, 
                                                     resource_attribute='K', position_attribute='position', 
                                                     condition_count=None, condition=None,  
                                                     update_count_agent=True, shuffle=True):   
        """
        Select some agents using the formula defined in https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/j.2006.0030-1299.15061.x
        I.E. the probability for a given agent to be selected is given by

                            D / (1 + exp(-a * (nb_agent_on_vertex/K_vertex - b)))

        Wher nb_agents_on_vertex is the number of agent on the vertex of the tested agent (this number is updated along the process), 
        K_vertex is the K parameter of the current vertex, and a, b and D are user provided parameters. 

        This methodology is typically applied to pick agents that should start a dispersion in ecological models.


        :param D: float
        :param beta: float
        :param alpha: float
        :param resource_attribute: optional, string, default 'K'. 
        :param position_attribute: optional, string, default 'position'.
        :param condition_count: optional, arr of bool, default None. Array telling which agent are to be counted.
        :param condition: optional, arr of bool, default None. Array telling which agent are susceptible to be picked
        :param update_count_agent: optional, boolean, default 'True'. If True, the number of agents per vertex is updated
            along the process (i.e. the number of agents is decreased by one each time an agent is picked.)
        :param shuffle: optional, boolean, default 'True'.
        """     
        if shuffle:
            perm = self.df_population.scramble(return_permutation=True)
            if condition is not None:
                condition = condition[perm]
            if condition_count is not None:
                condition_count = condition_count[perm]

        if condition_count is None:
            count_agents = count_nb_agent_per_vertex(self.df_population[position_attribute],
                                             self.graph.weights.shape[0])
        else:
            if (not isinstance(condition_count, np.ndarray)) or \
               (len(condition_count.shape) != 1) or \
               (condition_count.shape[0] != self.df_population.nb_rows) or \
               (not str(condition_count.dtype).startswith('bool')):
                raise ValueError("if used, condition argument should be a 1D array of bool of same length as the number"
                                 " of individuals.")
            count_agents = conditional_count_nb_agent_per_vertex(condition_count,
                                                         self.df_population[position_attribute],
                                                         self.graph.weights.shape[0])
        if update_count_agent:
            if condition is None:
                arr_rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                return movement_criterion_density_logistic_update_count(D, beta, alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand)
            else:
                arr_rand = np.random.uniform(0, 1, condition.sum())  
                return movement_criterion_density_logistic_update_count_conditional(D, beta, alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand, condition)

                          
        else:
            if condition is None:
                arr_rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                return movement_criterion_density_logistic(D, beta, alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand)
            else:
                arr_rand = np.random.uniform(0, 1, condition.sum()) 
                return movement_criterion_density_logistic_conditional(D, beta, alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand, condition)
        

    def pick_agents_using_density_resources_logistic_with_agents_category(self, arr_category, list_D, list_beta, list_alpha,                                                                           
                                                                        resource_attribute='K', position_attribute='position', 
                                                                        condition_count=None, condition=None,  
                                                                        update_count_agent=True, shuffle=True):   
        """
        Same as 'pick_agents_using_density_resources_logistic', but adapted to work simultaneously with a population divided into 
        categories. This method allows to avoid huge biases that would appear if pick_agents_using_density_resources_logistic was 
        being called sequentially on distinct category (the category subjected to the test first would be more susceptible to 
        be picked).

        Select some agents using the formula defined in https://nsojournals.onlinelibrary.wiley.com/doi/10.1111/j.2006.0030-1299.15061.x
        I.E. the probability for a given agent to be selected is given by

                            D / (1 + exp(-a * (nb_agent_on_vertex/K_vertex - b)))

        Where nb_agents_on_vertex is the number of agent on the vertex of the tested agent (this number is updated along the process), 
        K_vertex is the K parameter of the current vertex, and a, b and D are user provided parameters. 

        This methodology is typically applied to pick agents that should start a dispersion in ecological models.

        :param arr_category: 1D array of int with values in [0,n-1] for n categories.
        :param list_D: list of float of length n (n being the nb of categories)
        :param list_beta: list of float of length n (n being the nb of categories)
        :param list_alpha: list of float of length n (n being the nb of categories)
        :param resource_attribute: optional, string, default 'K'. 
        :param position_attribute: optional, string, default 'position'.
        :param condition_count: optional, arr of bool, default None. Array telling which agent are to be counted.
        :param condition: optional, arr of bool, default None. Array telling which agent are susceptible to be picked
        :param update_count_agent: optional, boolean, default 'True'. If True, the number of agents per vertex is updated
            along the process (i.e. the number of agents is decreased by one each time an agent is picked.)
        :param shuffle: optional, boolean, default 'True'.
        """    
        arr_D = np.array(list_D)
        arr_beta = np.array(list_beta)
        arr_alpha = np.array(list_alpha)

        if shuffle:
            perm = self.df_population.scramble(return_permutation=True)
            arr_category = arr_category[perm]
            if condition is not None:
                condition = condition[perm]
            if condition_count is not None:
                condition_count = condition_count[perm]

        if condition_count is None:
            count_agents = count_nb_agent_per_vertex(self.df_population[position_attribute],
                                             self.graph.weights.shape[0])
        else:
            if (not isinstance(condition_count, np.ndarray)) or \
               (len(condition_count.shape) != 1) or \
               (condition_count.shape[0] != self.df_population.nb_rows) or \
               (not str(condition_count.dtype).startswith('bool')):
                raise ValueError("if used, condition argument should be a 1D array of bool of same length as the number"
                                 " of individuals.")
            count_agents = conditional_count_nb_agent_per_vertex(condition_count,
                                                         self.df_population[position_attribute],
                                                         self.graph.weights.shape[0])
            
        if update_count_agent:
            if condition is None:
                arr_rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                return movement_criterion_density_logistic_update_count_with_agents_category(arr_category, arr_D, arr_beta, arr_alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand)
            else:
                arr_rand = np.random.uniform(0, 1, condition.sum())  
                return movement_criterion_density_logistic_update_count_conditional_with_agents_category(arr_category, arr_D, arr_beta, arr_alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand, condition)

                          
        else:
            if condition is None:
                arr_rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                return movement_criterion_density_logistic_with_agents_category(arr_category, arr_D, arr_beta, arr_alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand)
            else:
                arr_rand = np.random.uniform(0, 1, condition.sum()) 
                return movement_criterion_density_logistic_conditional_with_agents_category(arr_category, arr_D, arr_beta, arr_alpha, count_agents, 
                                                                        self.graph.df_attributes[resource_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_rand, condition)



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


# class TerritorialMovementWithResistance:
#     """
#     Add graph based movements abilities to the agents. Agents have both a territory and a position, which can be
#     different. Generally, the position of an agent will be either its territory vertex, or a vertex neighbouring its
#     territory. Here each connection in the graph is assumed to come with probability saying how likely a movement on
#     that connection is to fail.

#     Note that both position and territory are stored as integers, i.e. using the vertices indices which generally
#     differ from their id. If needed, conversion should be performed by the user using the graph attribute
#     'dict_cell_id_to_ind'. Equivalently, the user may extract a table giving the correspondence between id and indexes
#     using the graph method 'save_table_id_of_vertices_to_indices'.
#     """
#     def __init__(self, **kwargs):
#         if not hasattr(self, 'df_population'):
#             self.df_population = DataFrameXS()

#         self.df_population['territory'] = None
#         self.df_population['position'] = None

#     def _sampy_debug_change_territory_with_resistance(self,
#                                                       resistance_array,
#                                                       condition=None,
#                                                       territory_attribute='territory',
#                                                       position_attribute='position'):
#         if self.df_population.nb_rows == 0:
#             return
#         if (not isinstance(resistance_array, np.ndarray) or \
#                 resistance_array.shape != self.graph.connections.shape or \
#                 (not str(resistance_array.dtype).startswith('float'))):
#             raise ValueError("The resistance array should be a 2 dimensional array of floats of shape " +
#                              str(resistance_array.shape))
#         if condition is not None:
#             if (not isinstance(condition, np.ndarray)) or \
#                     (condition.shape != (self.df_population.nb_rows,)) or \
#                     (not str(condition.dtype).startswith('bool')):
#                 raise ValueError("if used, condition argument should be a 1D array of bool of same length as the number"
#                                  " of individuals.")

#         check_col_exists_good_type(self.df_population, territory_attribute, prefix_dtype='int')
#         check_col_exists_good_type(self.df_population, position_attribute, prefix_dtype='int')

#     def change_territory_with_resistance(self,
#                                          resistance_array,
#                                          condition=None,
#                                          territory_attribute='territory',
#                                          position_attribute='position'):
#         """
#         Change the territory and the position of the agents. If an agent is on an isolated vertex (a vertex without
#         any neighbour), then the agent stays on the vertex.

#         :param resistance_array: 2d array of float, array of same shape as the connections of the graph, gives the
#                                  'resistance to movement' of each connection
#         :param condition: optional, array of bool, default None. If not None, array telling which
#         :param territory_attribute: optional, string, default 'territory'
#         :param position_attribute: optional, string, default 'position'
#         """
#         if self.df_population.nb_rows == 0:
#             return
#         if condition is not None:
#             rand = np.random.uniform(0, 1, (condition.sum(),))
#             movement_change_territory_and_position_condition(self.df_population[territory_attribute],
#                                                              self.df_population[position_attribute],
#                                                              condition, rand,
#                                                              self.graph.connections, self.graph.weights)
#         else:
#             rand = np.random.uniform(0, 1, (self.df_population.shape[0],))
#             movement_change_territory_and_position(self.df_population[territory_attribute],
#                                                    self.df_population[position_attribute],
#                                                    rand, self.graph.connections, self.graph.weights)

#     def _sampy_debug_mov_around_territory(self,
#                                           proba_remain_on_territory,
#                                           condition=None,
#                                           territory_attribute='territory',
#                                           position_attribute='position'):
#         if self.df_population.nb_rows == 0:
#             return
#         if condition is not None:
#             check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))
#         check_col_exists_good_type(self.df_population, territory_attribute, 'territory_attribute', prefix_dtype='int',
#                                    reject_none=True)
#         check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', prefix_dtype='int',
#                                    reject_none=True)

#     def mov_around_territory(self,
#                              proba_remain_on_territory,
#                              condition=None,
#                              territory_attribute='territory',
#                              position_attribute='position'):
#         """
#         Update the average position of the agent around its territory during the current time step.

#         :param proba_remain_on_territory: float, probability to stay on the territory
#         :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
#                           True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
#                           move.
#         :param territory_attribute: optional, string, default 'territory'
#         :param position_attribute: optional, string, default 'position'
#         """
#         if self.df_population.nb_rows == 0:
#             return
#         if condition is not None:
#             pre_bool_mov = np.random.uniform(0, 1, condition.sum()) > proba_remain_on_territory
#             bool_mov = movement_mov_around_territory_fill_bool_mov_using_condition(pre_bool_mov, condition)
#         else:
#             bool_mov = np.random.uniform(0, 1, self.df_population.shape[0]) > proba_remain_on_territory
#         rand = np.random.uniform(0, 1, bool_mov.sum())
#         movement_mov_around_territory(self.df_population[territory_attribute], self.df_population[position_attribute],
#                                       bool_mov, rand, self.graph.connections, self.graph.weights)

#     def _sampy_debug_dispersion_with_varying_nb_of_steps(self, arr_nb_steps, arr_prob,
#                                                          condition=None,
#                                                          territory_attribute='territory',
#                                                          position_attribute='position'
#                                                          ):
#         if self.df_population.nb_rows == 0:
#             return

#         check_input_array(arr_nb_steps, 'arr_nb_steps', 'int', nb_dim=1)
#         check_input_array(arr_prob, 'arr_prob', 'float', nb_dim=1)

#         if arr_prob.shape != arr_nb_steps.shape:
#             raise ValueError("Arguments 'arr_nb_steps' and 'arr_prob' have different shapes.")

#         if condition is not None:
#             check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

#         check_col_exists_good_type(self.df_population, territory_attribute, 'territory_attribute', prefix_dtype='int',
#                                    reject_none=True)
#         check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', prefix_dtype='int',
#                                    reject_none=True)

#     def dispersion_with_varying_nb_of_steps(self, arr_nb_steps, arr_prob,
#                                             condition=None,
#                                             territory_attribute='territory',
#                                             position_attribute='position'
#                                             ):
#         """
#         Used to modelize dispersion of agents. Each selected agent will perform a random number of discrete steps on
#         the graph. The number of steps is determined using the user inputs 'arr_nb_steps' and 'arr_prob'. Both
#         position and territory are updated.

#         :param arr_nb_steps: 1D array of int, giving the permissible number of steps.
#         :param arr_prob: 1D array of float, arr_prob[i] is the probability that a given agent will perform
#                          arr_nb_steps[i] steps.
#         :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
#                           True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
#                           move. If left at None, all the agents move.
#         :param territory_attribute: optional, string, default 'territory'.
#         :param position_attribute: optional, string, default 'position'.
#         """
#         if self.df_population.nb_rows == 0:
#             return
#         prob = arr_prob.astype('float64')
#         prob = prob / prob.sum()
#         if condition is not None:
#             # get number of steps
#             arr_nb_steps = np.random.choice(arr_nb_steps, condition.sum(), p=prob)
#         else:
#             arr_nb_steps = np.random.choice(arr_nb_steps, self.df_population.nb_rows, p=prob)
#         rand = np.random.uniform(0, 1, arr_nb_steps.sum())
#         if condition is None:
#             movement_dispersion_with_varying_nb_of_steps(self.df_population[territory_attribute],
#                                                          self.df_population[position_attribute],
#                                                          rand, arr_nb_steps, self.graph.connections, self.graph.weights)
#         else:
#             movement_dispersion_with_varying_nb_of_steps_condition(self.df_population[territory_attribute],
#                                                                    self.df_population[position_attribute],
#                                                                    condition,
#                                                                    rand, arr_nb_steps, self.graph.connections,
#                                                                    self.graph.weights)


class TerritorialDirectionalMovementWithoutResistance:
    """
    TODO : work only with Oriented hexagonal grid or graphs derived from it
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

        self.df_population['territory'] = None
        self.df_population['position'] = None
        self.add_attribute('direction', -1)

    def directional_dispersion_from_arr_nb_steps(self, 
                                                 arr_nb_steps,
                                                 arr_directional_prob,                             
                                                 condition=None,
                                                 territory_attribute='territory',
                                                 position_attribute='position',
                                                 direction_attribute='direction',
                                                 reinitialize_direction=False,
                                                 rebound=True,
                                                 return_path=False,
                                                 update_dir=False):
        """
        TODO
        Used to modelize dispersion of agents. Each selected agent will perform a number of steps on
        the graph. The number of steps is determined using the user inputs 'arr_nb_steps' and 'arr_prob'. Both
        position and territory are updated.

        :param arr_nb_steps: 1D array of int, giving the permissible number of steps.
        :param arr_directional_prob: 1D array of float, arr_directional_prob[i] is the probability that a given agent will perform
                         arr_nb_steps[i] steps. Note that arr_directional_prob[0] is the probability ...
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move. If left at None, all the agents move.
        :param territory_attribute: optional, string, default 'territory'.
        :param position_attribute: optional, string, default 'position'.
        """

        if self.df_population.nb_rows == 0:
            return

        arr_prob = arr_directional_prob.astype('float64')
        arr_cumul_directional_prob = np.cumsum(arr_prob)
        arr_cumul_directional_prob = arr_cumul_directional_prob / arr_cumul_directional_prob[-1]
        arr_cumul_directional_prob[-1] = 1.

        if reinitialize_direction:
           self.df_population[direction_attribute] = -1 

        if condition is not None:
            arr_nb_steps = arr_nb_steps * condition


        if rebound:
            max_nb_rand = 2 * arr_nb_steps.sum()
            arr_rand = np.random.uniform(0, 1, max_nb_rand)

            if return_path:
                return movement_directional_dispersion_with_varying_nb_of_steps_return_path(self.df_population[territory_attribute], 
                                                                        self.df_population[position_attribute], 
                                                                        arr_nb_steps, arr_cumul_directional_prob,
                                                                        self.graph.connections, self.graph.weights,
                                                                        arr_rand, self.df_population[direction_attribute], self.df_population['col_id'], update_dir)
            
            movement_directional_dispersion_with_varying_nb_of_steps(self.df_population[territory_attribute], 
                                                                     self.df_population[position_attribute], 
                                                                     arr_nb_steps, arr_cumul_directional_prob,
                                                                     self.graph.connections, self.graph.weights,
                                                                     arr_rand, self.df_population[direction_attribute])
            
        else:
            max_nb_rand = arr_nb_steps.sum()
            arr_rand = np.random.uniform(0, 1, max_nb_rand)
            raise NotImplementedError
