import numpy as np
import pandas as pd
import time

from .jit_compiled_functions import (random_walk_on_sphere_set_position_based_on_graph,
                                     random_walk_on_sphere_start_random_walk_uniform_prob,
                                     conditional_random_walk_on_sphere_start_random_walk_uniform_prob,
                                     random_walk_on_sphere_set_initial_dir_to_north,
                                     random_walk_on_sphere_deviate_direction,
                                     random_walk_propose_step_gamma_law,
                                     _temp_random_walk_on_sphere_exit_random_walk_based_on_k)

# ---------------------------------------------------------------------------------------------------------------------
# This section of Sampy is still a work in progress and will have its own dedicated publication introducing the
# "spherical random walk" coded here. In the future, more usual options like correlated and biased 2D random walks will
# be added here as well.
# ---------------------------------------------------------------------------------------------------------------------


class ProposedStep:
    """
    This class contains the information concerning a proposed random walk step. It is returned by the methods that
    propose a new step for a randomly walking agent. It includes methods to check if the steps are valid with respect
    to proximity objects.
    """
    def __init__(self, agent_object, arr_new_steps, pos_x, pos_y, pos_z, dir_x, dir_y, dir_z):
        self.agent = agent_object
        self.arr_new_steps = arr_new_steps
        self.pos_x = pos_x
        self.pos_y = pos_y
        self.pos_z = pos_z
        self.dir_x = dir_x
        self.dir_y = dir_y
        self.dir_z = dir_z

    def get_array_remaining_steps(self):
        return self.arr_new_steps

    def is_step_allowed(self, proximity_class, condition_on_grid=None):
        """
        Check if the currently stored steps are valid with respect to a given proximity class
        :param proximity_class: proximity class object
        :param condition_on_grid: optional, 1D array of bool, default None. Add a condition on the nodes of the grid
            used by the proximity class.
        :return: array of boolean telling which steps are allowed
        """
        return proximity_class.is_step_allowed(self.arr_new_steps, self.pos_x, self.pos_y, self.pos_z,
                                               condition_on_grid=condition_on_grid)

    def validate_steps(self, arr_accepted_steps, update_arr_new_steps=True):
        """
        Change the coordinates and direction of the agents whose steps have been accepted
        :param arr_accepted_steps: 1D array of bool saying which agent have their step validated
        :param update_arr_new_steps: optional, bool, default True. If set to True, the method will replace the attribute
            arr_new_steps with a 1D array of bool saying which agent did not have their step validated. This especially
            useful if one wants to check which steps are valid with respect to several successive proximity classes.
        """
        self.agent.df_population['coord_x'] = self.agent.df_population['coord_x'] + \
                                              arr_accepted_steps * (self.pos_x - self.agent.df_population['coord_x'])
        self.agent.df_population['coord_y'] = self.agent.df_population['coord_y'] + \
                                              arr_accepted_steps * (self.pos_y - self.agent.df_population['coord_y'])
        self.agent.df_population['coord_z'] = self.agent.df_population['coord_z'] + \
                                              arr_accepted_steps * (self.pos_z - self.agent.df_population['coord_z'])
        self.agent.df_population['direction_x'] = self.agent.df_population['direction_x'] + \
                                            arr_accepted_steps * (self.dir_x - self.agent.df_population['direction_x'])
        self.agent.df_population['direction_y'] = self.agent.df_population['direction_y'] + \
                                            arr_accepted_steps * (self.dir_y - self.agent.df_population['direction_y'])
        self.agent.df_population['direction_z'] = self.agent.df_population['direction_z'] + \
                                            arr_accepted_steps * (self.dir_z - self.agent.df_population['direction_z'])
        if update_arr_new_steps:
            self.arr_new_steps = self.arr_new_steps & ~arr_accepted_steps


class RandomWalkOnSphere:
    """
    This class give the ability for the agent to perform random walks on a sphere. The directions are preserved between
    consecutive steps by using parallel transport.
    """
    def __init__(self):
        self.radius = 1.
        self.unit = 'km'
        if hasattr(self, 'df_population'):
            self.df_population['is_on_random_walk'] = False
            self.df_population['coord_x'] = np.nan
            self.df_population['coord_y'] = np.nan
            self.df_population['coord_z'] = np.nan
            self.df_population['direction_x'] = np.nan
            self.df_population['direction_y'] = np.nan
            self.df_population['direction_z'] = np.nan
        else:
            self.df_population = pd.DataFrame(columns=['is_on_random_walk', 'coord_x', 'coord_y', 'coord_z',
                                                       'direction_x', 'direction_y', 'direction_z'])

        if hasattr(self, 'dict_default_values'):
            self.dict_default_values['is_on_random_walk'] = False
            self.dict_default_values['coord_x'] = np.nan
            self.dict_default_values['coord_y'] = np.nan
            self.dict_default_values['coord_z'] = np.nan
            self.dict_default_values['direction_x'] = np.nan
            self.dict_default_values['direction_y'] = np.nan
            self.dict_default_values['direction_z'] = np.nan
        else:
            self.dict_default_values = dict()
            self.dict_default_values['is_on_random_walk'] = False
            self.dict_default_values['coord_x'] = np.nan
            self.dict_default_values['coord_y'] = np.nan
            self.dict_default_values['coord_z'] = np.nan
            self.dict_default_values['direction_x'] = np.nan
            self.dict_default_values['direction_y'] = np.nan
            self.dict_default_values['direction_z'] = np.nan
        super().__init__()

    def set_radius(self, radius, unit='km'):
        """
        Set the value of the radius of the sphere on which the agents walk.
        :param radius: float, value for the radius of the sphere.
        :param unit: optional, string, default 'km'. Unit used for distances. For the moment, this parameter is not
            used by any method.
        """
        self.radius = radius
        self.unit = unit

    def set_position_based_on_graph(self, arr_selected_agent, use_radius=True, agent_position_attribute='position',
                                    graph_coord_x_attribute='coord_x',
                                    graph_coord_y_attribute='coord_y',
                                    graph_coord_z_attribute='coord_z'):
        """
        Set the position of the selected agents using coordinates of the graph vertex the agent is currently on.
        :param arr_selected_agent: 1D array of bool, saying which agents should have their coordinates updated according
            to their position on the graph.
        :param use_radius: Optional, boolean, default True.
        :param agent_position_attribute: option, string, default 'position'. Name of the attribute position column in
            the df_population dataframe in the agent class.
        :param graph_coord_x_attribute:
        :param graph_coord_y_attribute:
        :param graph_coord_z_attribute:
        """
        arr_selected_agent = np.array(arr_selected_agent, dtype=bool)
        arr_pos_agent = np.array(self.df_population[agent_position_attribute], dtype=np.int32)
        coord_x = np.array(self.df_population['coord_x'], dtype=np.float)
        coord_y = np.array(self.df_population['coord_y'], dtype=np.float)
        coord_z = np.array(self.df_population['coord_z'], dtype=np.float)
        graph_coord_x = np.array(self.graph.df_attributes[graph_coord_x_attribute], dtype=np.float)
        graph_coord_y = np.array(self.graph.df_attributes[graph_coord_y_attribute], dtype=np.float)
        graph_coord_z = np.array(self.graph.df_attributes[graph_coord_z_attribute], dtype=np.float)
        x, y, z = random_walk_on_sphere_set_position_based_on_graph(arr_selected_agent, arr_pos_agent, coord_x, coord_y,
                                                                    coord_z, graph_coord_x, graph_coord_y, graph_coord_z)
        if use_radius:
            self.df_population['coord_x'] = self.df_population['coord_x'] + arr_selected_agent * \
                                            (self.radius * x - self.df_population['coord_x'])
            self.df_population['coord_y'] = self.df_population['coord_y'] + arr_selected_agent * \
                                            (self.radius * y - self.df_population['coord_y'])
            self.df_population['coord_z'] = self.df_population['coord_z'] + arr_selected_agent * \
                                            (self.radius * z - self.df_population['coord_z'])
        else:
            self.df_population['coord_x'] = self.df_population['coord_x'] + arr_selected_agent * \
                                            (x - self.df_population['coord_x'])
            self.df_population['coord_y'] = self.df_population['coord_y'] + arr_selected_agent * \
                                            (y - self.df_population['coord_y'])
            self.df_population['coord_z'] = self.df_population['coord_z'] + arr_selected_agent * \
                                            (z - self.df_population['coord_z'])

    def start_random_walk_uniform_prob(self, prob, condition=None):
        """
        Perform a uniform test for eahc agent. Successful agents have their attribute 'is_on_random_walk' set to true.
        :param prob: float, probability for all agent to start a random walk
        :param condition: optional, 1D array of bool, default None. If not None, the size of this argument should be the
            total number of agents, and the agent in row i will perform the test if and only if condition[i] is True.
        :return: 1D array of bool telling which agents started random walk thanks to this method (excluding
        """
        arr_is_on_rw = np.array(self.df_population['is_on_random_walk'], dtype=bool)
        if condition is None:
            arr_start_rw = np.random.uniform(0, 1, (self.df_population.shape[0],)) <= prob
            old = np.array(self.df_population['is_on_random_walk'].copy(), dtype=np.bool_)
            new = random_walk_on_sphere_start_random_walk_uniform_prob(arr_start_rw, arr_is_on_rw)
        else:
            condition = np.array(condition, dtype=bool)
            arr_start_rw = np.random.uniform(0, 1, (condition.sum(),)) <= prob
            old = np.array(self.df_population['is_on_random_walk'].copy(), dtype=np.bool_)
            new = conditional_random_walk_on_sphere_start_random_walk_uniform_prob(arr_start_rw, arr_is_on_rw, condition)
        self.df_population['is_on_random_walk'] = new
        return new & ~old

    def _temp_exit_random_walk_based_on_k(self, arr_selected_agents, arr_pos_selected_agents, prob_settlement, alpha,
                                          arr_pop=None):
        """
        This method is considered private as it is a quick and dirty hack to have a 'kind of realistic' condition for
        exiting random walk state. This is not based on ANY literature, so be careful and use at your own risk.
        """
        arr_selected_agents = np.array(arr_selected_agents)
        # print('selected_agents', arr_selected_agents)
        arr_pos_selected_agents = np.array(arr_pos_selected_agents, dtype=np.int32)
        # print('pos_agents', arr_pos_selected_agents)
        # print(arr_pos_selected_agents)
        nb_agents = arr_selected_agents.sum()
        rand = np.random.uniform(0, 1, (nb_agents,))
        arr_k = np.array(self.graph.df_attributes['K'])
        if arr_pop is None:
            arr_pop = self.count_pop_per_vertex(position_attribute='territory')
        arr_pop = np.array(arr_pop, dtype=np.int32)
        arr_stop = _temp_random_walk_on_sphere_exit_random_walk_based_on_k(arr_selected_agents, rand, prob_settlement,
                                                                           alpha, arr_pos_selected_agents, arr_k,
                                                                           arr_pop)
        # print('stopping agents:', arr_stop)
        self.df_population['is_on_random_walk'] = self.df_population['is_on_random_walk'] & ~arr_stop
        # print(self.df_population[['territory', 'col_id']])
        # print(self.df_population['territory'])
        self.df_population['territory'] = self.df_population['territory'] + \
                                          arr_stop * (arr_pos_selected_agents - self.df_population['territory'])
        # print(' ')
        # print(self.df_population['territory'])
        # print('---------')
        # time.sleep(2.)

        self.df_population['position'] = self.df_population['position'] + \
                                          arr_stop * (arr_pos_selected_agents - self.df_population['position'])
        return arr_stop

    def set_direction_to_north(self, arr_selected_agents):
        """
        Set the directions of the selected agents in the direction of the north pole (coordinates X et Y are 0.). Agents
        located at the north or south pole have their initial direction set to (1., 0., 0.).
        :param arr_selected_agents:
        """
        arr_selected_agents = np.array(arr_selected_agents, dtype=bool)

        pos_x = np.array(self.df_population['coord_x'], dtype=np.float)
        pos_y = np.array(self.df_population['coord_y'], dtype=np.float)
        pos_z = np.array(self.df_population['coord_z'], dtype=np.float)

        dir_x = np.array(self.df_population['direction_x'], dtype=np.float)
        dir_y = np.array(self.df_population['direction_y'], dtype=np.float)
        dir_z = np.array(self.df_population['direction_z'], dtype=np.float)

        dir_x, dir_y, dir_z = random_walk_on_sphere_set_initial_dir_to_north(arr_selected_agents, pos_x, pos_y,
                                                                             pos_z, dir_x, dir_y, dir_z)

        self.df_population['direction_x'] = dir_x
        self.df_population['direction_y'] = dir_y
        self.df_population['direction_z'] = dir_z

    def set_direction_von_mises(self, arr_selected_agents, kappa):
        """
        Set the direction of the selected agents by deviating their current direction by an angle given by von mises
        distribution.
        :param arr_selected_agents: 1D array of bool saying which agent should have their direction changed
        :param kappa: kappa parameter for the von mises distribution.
        """
        arr_selected_agents = np.array(arr_selected_agents, dtype=bool)

        pos_x = np.array(self.df_population['coord_x'], dtype=np.float)
        pos_y = np.array(self.df_population['coord_y'], dtype=np.float)
        pos_z = np.array(self.df_population['coord_z'], dtype=np.float)

        dir_x = np.array(self.df_population['direction_x'], dtype=np.float)
        dir_y = np.array(self.df_population['direction_y'], dtype=np.float)
        dir_z = np.array(self.df_population['direction_z'], dtype=np.float)

        deviation_angles = np.random.vonmises(0, kappa, (arr_selected_agents.sum(),))
        dir_x, dir_y, dir_z = random_walk_on_sphere_deviate_direction(deviation_angles, arr_selected_agents,
                                                                      pos_x, pos_y, pos_z, dir_x, dir_y, dir_z)

        self.df_population['direction_x'] = dir_x
        self.df_population['direction_y'] = dir_y
        self.df_population['direction_z'] = dir_z

    def propose_step_gamma_law(self, arr_selected_agents, k, theta):
        """
        Propose a new step for the selected agents, following their current direction with a step length given by a
        Gamma distribution.
        :param arr_selected_agents: 1D array of bool saying which agent should make a new step
        :param k: shape parameter gamma law
        :param theta: scale parameter gamma law
        :return: ProposedStep class
        """
        gamma_sample = np.random.gamma(k, theta, size=(arr_selected_agents.sum(),))

        arr_selected_agents = np.array(arr_selected_agents)

        pos_x = np.array(self.df_population['coord_x'], dtype=np.float)
        pos_y = np.array(self.df_population['coord_y'], dtype=np.float)
        pos_z = np.array(self.df_population['coord_z'], dtype=np.float)

        dir_x = np.array(self.df_population['direction_x'], dtype=np.float)
        dir_y = np.array(self.df_population['direction_y'], dtype=np.float)
        dir_z = np.array(self.df_population['direction_z'], dtype=np.float)

        px, py, pz, dx, dy, dz = random_walk_propose_step_gamma_law(arr_selected_agents, gamma_sample, pos_x,
                                                                    pos_y, pos_z, dir_x, dir_y, dir_z, self.radius)

        return ProposedStep(self, arr_selected_agents, px, py, pz, dx, dy, dz)