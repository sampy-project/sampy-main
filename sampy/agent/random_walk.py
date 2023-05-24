import numpy as np
import pandas as pd
import time

from .jit_compiled_functions import (random_walk_on_sphere_set_position_based_on_graph,
                                     random_walk_on_sphere_start_random_walk_uniform_prob,
                                     random_walk_on_sphere_start_random_walk_uniform_prob_return_new_walkers,
                                     conditional_random_walk_on_sphere_start_random_walk_uniform_prob,
                                     conditional_random_walk_on_sphere_start_random_walk_uniform_prob_return_new_walkers,
                                     random_walk_on_sphere_set_initial_dir_to_north,
                                     random_walk_on_sphere_deviate_direction_from_angles,
                                     random_walk_on_sphere_propose_step_gamma_law,
                                     random_walk_on_sphere_make_step_gamma_law,
                                     random_walk_on_sphere_validate_step,
                                     random_walk_on_sphere_validate_step_return_fail,
                                     _temp_random_walk_on_sphere_exit_random_walk_based_on_k)


class SphericalRandomWalk:
    """
    This class give the ability for the agent to perform random walks on a sphere. The directions are preserved between
    consecutive steps by using parallel transport. The associated technical background can be found in the publication
    {in-preparation} by F. Viard, A. Allibert and P. Leighton.
    """
    def __init__(self, px=0., py=0., pz=0., dx=0., dy=0., dz=0., **kwargs):
        self.df_population['is_on_random_walk'] = False
        self.df_population['px'] = px
        self.df_population['py'] = py
        self.df_population['pz'] = pz
        self.df_population['dx'] = dx
        self.df_population['dy'] = dy
        self.df_population['dz'] = dz

        self.dict_default_values['is_on_random_walk'] = False
        self.dict_default_values['px'] = px
        self.dict_default_values['py'] = py
        self.dict_default_values['pz'] = pz
        self.dict_default_values['dx'] = dx
        self.dict_default_values['dy'] = dy
        self.dict_default_values['dz'] = dz

    def set_dtype_of_positions_and_directions(self, targeted_dtype='float64'):
        """
        The computations needed for spherical random walks, as they are now, are really sensitive to rounding errors.
        Therefore, it may be needed to increase the precision of columns px, py, pz, dx, dy and dz to float64. This
        method is a shortcut for doing it on all 6 columns. This method won't have any effect if there is no agents.

        :param targeted_dtype: optional, string, default 'float64'. The new datatype for the position and direction
                               columns.

        :return: True if the columns were not empty and the assignement succeded, False otherwise.
        """
        if self.df_population.is_empty:
            return False

        self.df_population.change_type('px', targeted_dtype)
        self.df_population.change_type('py', targeted_dtype)
        self.df_population.change_type('pz', targeted_dtype)
        self.df_population.change_type('dx', targeted_dtype)
        self.df_population.change_type('dy', targeted_dtype)
        self.df_population.change_type('dz', targeted_dtype)

        return True

    def set_position_based_on_graph(self, arr_selected_agent,
                                    agent_position_attribute='position',
                                    graph_coord_x_attribute='coord_x',
                                    graph_coord_y_attribute='coord_y',
                                    graph_coord_z_attribute='coord_z'):
        """
        Set the position of the selected agents using coordinates of the graph vertex the agent is currently on.

        :param arr_selected_agent: 1D array of bool, saying which agents should have their coordinates updated according
                                   to their position on the graph.
        :param agent_position_attribute: optional, string, default 'position'. Name of the attribute position column in
                                         the df_population dataframe in the agent class.
        :param graph_coord_x_attribute: optional, string, default 'coord_x'. X coordinate of the graph vertices.
        :param graph_coord_y_attribute: optional, string, default 'coord_y'. Y coordinate of the graph vertices.
        :param graph_coord_z_attribute: optional, string, default 'coord_z'. Z coordinate of the graph vertices.
        """
        random_walk_on_sphere_set_position_based_on_graph(arr_selected_agent,
                                                          self.df_population[agent_position_attribute],
                                                          self.df_population['px'],
                                                          self.df_population['py'],
                                                          self.df_population['pz'],
                                                          self.graph.df_attributes[graph_coord_x_attribute],
                                                          self.graph.df_attributes[graph_coord_y_attribute],
                                                          self.graph.df_attributes[graph_coord_z_attribute])

    def start_random_walk_uniform_prob(self, prob, condition=None, return_arr_new_walker=True):
        """
        Perform a uniform test for each agent. Successful agents have their attribute 'is_on_random_walk' set to true.

        :param prob: float, probability for all agent to start a random walk
        :param condition: optional, 1D array of bool, default None.
        :param return_arr_new_walker: optional, boolean, default True.

        :return: None if return_arr_new_walker is False, a 1D array of bool otherwise.
        """
        if condition is None:
            arr_start_rw = np.random.uniform(0, 1, (self.df_population.shape[0],)) <= prob
            if return_arr_new_walker:
                return random_walk_on_sphere_start_random_walk_uniform_prob_return_new_walkers(arr_start_rw,
                                                                                self.df_population['is_on_random_walk'])
            random_walk_on_sphere_start_random_walk_uniform_prob(arr_start_rw, self.df_population['is_on_random_walk'])
        else:
            arr_start_rw = np.random.uniform(0, 1, (condition.sum(),)) <= prob
            if return_arr_new_walker:
                return conditional_random_walk_on_sphere_start_random_walk_uniform_prob_return_new_walkers(arr_start_rw,
                                                                                                           self.df_population['is_on_random_walk'],
                                                                                                           condition)
            conditional_random_walk_on_sphere_start_random_walk_uniform_prob(arr_start_rw,
                                                                             self.df_population['is_on_random_walk'],
                                                                             condition)

    def set_direction_to_north(self, arr_selected_agents):
        """
        Set the directions of the selected agents in the direction of the north pole (coordinates X and Y are 0.).
        Agents located at the north or south pole have their initial direction set to (1., 0., 0.). Note that the
        direction is a vector of norm 1.

        :param arr_selected_agents: 1D array of bool
        """
        random_walk_on_sphere_set_initial_dir_to_north(arr_selected_agents,
                                                       self.df_population['px'],
                                                       self.df_population['py'],
                                                       self.df_population['pz'],
                                                       self.df_population['dx'],
                                                       self.df_population['dy'],
                                                       self.df_population['dz'])

    def set_direction_von_mises(self, arr_selected_agents, kappa):
        """
        Set the direction of the selected agents by deviating their current direction by an angle given by von mises
        distribution.

        :param arr_selected_agents: 1D array of bool saying which agent should have their direction changed
        :param kappa: kappa parameter for the von mises distribution. The hiher the value of Kappa, the smaller the
                      deviation.
        """
        deviation_angles = np.random.vonmises(0, kappa, (arr_selected_agents.sum(),))
        random_walk_on_sphere_deviate_direction_from_angles(deviation_angles, arr_selected_agents,
                                                            self.df_population['px'],
                                                            self.df_population['py'],
                                                            self.df_population['pz'],
                                                            self.df_population['dx'],
                                                            self.df_population['dy'],
                                                            self.df_population['dz'])

    def make_step_using_gamma_law(self, arr_selected_agents, k, theta, radius,
                                  list_proximity_classes=None, mode_proximity_test='AND',
                                  return_agents_failed_making_step=False):
        """
        The selected agents make a step, following their current direction with a step length given by a
        Gamma distribution. The user can provide a list of proximity class to test if the step fails or succeed.

        :param arr_selected_agents: 1D array of bool saying which agent should make a new step
        :param k: shape parameter gamma law
        :param theta: scale parameter gamma law
        :param radius: float, radius of the sphere on which the agents live
        :param list_proximity_classes: optional, list of proximity class, default None. If a list of proximity class is
                                       provided, the method will check that each step is valid according to those
                                       proximity class. If the list has more than one element, this test is done using
                                       the methodology defined by the kwarg mode_proximity_test.
        :param mode_proximity_test: optional, string, default 'AND'. Only two accepted values, that are 'AND' and 'OR'.
                                    If 'AND', a step is valid if and only if all the proximity classes validate it.
                                    IF 'OR', a step is valid if at least one of the proximity classes validate it.
        :param return_agents_failed_making_step: optional, boolean, default False. If True, returns a 1D array of bool
                                                 Saying which agent failed their step.
                                                 WARNING: in the resulting array res, res[i] is True if and only if
                                                          the agent at line i had the opportunity to make a step but
                                                          failed to do so. If res[i] is False, then it means the agent
                                                          at line i either was not selected to make a step, or succeded
                                                          to do so.

        :return: if return_agents_failed_making_step is True, 1D array of bool. None otherwise.
        """
        gamma_sample = np.random.gamma(k, theta, size=(arr_selected_agents.sum(),))

        if list_proximity_classes is not None:

            new_px, new_py, new_pz, new_dx, new_dy, new_dz = random_walk_on_sphere_propose_step_gamma_law(
                                                                                  arr_selected_agents, gamma_sample,
                                                                                  self.df_population['px'],
                                                                                  self.df_population['py'],
                                                                                  self.df_population['pz'],
                                                                                  self.df_population['dx'],
                                                                                  self.df_population['dy'],
                                                                                  self.df_population['dz'],
                                                                                  radius)

            list_arr_successful_steps = []
            for proximity_class in list_proximity_classes:
                list_arr_successful_steps.append(proximity_class.is_pos_allowed(new_px, new_py, new_pz))
            if mode_proximity_test.lower() == 'and':
                for i, arr in enumerate(list_arr_successful_steps):
                    if i == 0:
                        continue
                    list_arr_successful_steps[0] = list_arr_successful_steps[0] & arr
            elif mode_proximity_test.lower() == 'or':
                for i, arr in enumerate(list_arr_successful_steps):
                    if i == 0:
                        continue
                    list_arr_successful_steps[0] = list_arr_successful_steps[0] | arr
            else:
                raise ValueError("The value for mode_proximity_test kwarg should either be 'AND' or 'OR'.")

            if return_agents_failed_making_step:
                return random_walk_on_sphere_validate_step_return_fail(arr_selected_agents,
                                                                       list_arr_successful_steps[0],
                                                                       new_px, new_py, new_pz, new_dx, new_dy, new_dz,
                                                                       self.df_population['px'],
                                                                       self.df_population['py'],
                                                                       self.df_population['pz'],
                                                                       self.df_population['dx'],
                                                                       self.df_population['dy'],
                                                                       self.df_population['dz'])

            random_walk_on_sphere_validate_step(arr_selected_agents, list_arr_successful_steps[0],
                                                new_px, new_py, new_pz, new_dx, new_dy, new_dz,
                                                self.df_population['px'],
                                                self.df_population['py'],
                                                self.df_population['pz'],
                                                self.df_population['dx'],
                                                self.df_population['dy'],
                                                self.df_population['dz'])

        else:
            random_walk_on_sphere_make_step_gamma_law(arr_selected_agents, gamma_sample,
                                                      self.df_population['px'],
                                                      self.df_population['py'],
                                                      self.df_population['pz'],
                                                      self.df_population['dx'],
                                                      self.df_population['dy'],
                                                      self.df_population['dz'],
                                                      radius)


# class RandomWalkOnSphere:
#     """
#     This class give the ability for the agent to perform random walks on a sphere. The directions are preserved between
#     consecutive steps by using parallel transport.
#     """
#     def __init__(self):
#         self.radius = 1.
#         self.unit = 'km'
#         if hasattr(self, 'df_population'):
#             self.df_population['is_on_random_walk'] = False
#             self.df_population['coord_x'] = np.nan
#             self.df_population['coord_y'] = np.nan
#             self.df_population['coord_z'] = np.nan
#             self.df_population['direction_x'] = np.nan
#             self.df_population['direction_y'] = np.nan
#             self.df_population['direction_z'] = np.nan
#         else:
#             self.df_population = pd.DataFrame(columns=['is_on_random_walk', 'coord_x', 'coord_y', 'coord_z',
#                                                        'direction_x', 'direction_y', 'direction_z'])
#
#         if hasattr(self, 'dict_default_values'):
#             self.dict_default_values['is_on_random_walk'] = False
#             self.dict_default_values['coord_x'] = np.nan
#             self.dict_default_values['coord_y'] = np.nan
#             self.dict_default_values['coord_z'] = np.nan
#             self.dict_default_values['direction_x'] = np.nan
#             self.dict_default_values['direction_y'] = np.nan
#             self.dict_default_values['direction_z'] = np.nan
#         else:
#             self.dict_default_values = dict()
#             self.dict_default_values['is_on_random_walk'] = False
#             self.dict_default_values['coord_x'] = np.nan
#             self.dict_default_values['coord_y'] = np.nan
#             self.dict_default_values['coord_z'] = np.nan
#             self.dict_default_values['direction_x'] = np.nan
#             self.dict_default_values['direction_y'] = np.nan
#             self.dict_default_values['direction_z'] = np.nan
#         super().__init__()
#
#     def set_radius(self, radius, unit='km'):
#         """
#         Set the value of the radius of the sphere on which the agents walk.
#         :param radius: float, value for the radius of the sphere.
#         :param unit: optional, string, default 'km'. Unit used for distances. For the moment, this parameter is not
#             used by any method.
#         """
#         self.radius = radius
#         self.unit = unit
#
#     def _temp_exit_random_walk_based_on_k(self, arr_selected_agents, arr_pos_selected_agents, prob_settlement, alpha,
#                                           arr_pop=None):
#         """
#         This method is considered private as it is a quick and dirty hack to have a 'kind of realistic' condition for
#         exiting random walk state. This is not based on ANY literature, so be careful and use at your own risk.
#         """
#         arr_selected_agents = np.array(arr_selected_agents)
#         # print('selected_agents', arr_selected_agents)
#         arr_pos_selected_agents = np.array(arr_pos_selected_agents, dtype=np.int32)
#         # print('pos_agents', arr_pos_selected_agents)
#         # print(arr_pos_selected_agents)
#         nb_agents = arr_selected_agents.sum()
#         rand = np.random.uniform(0, 1, (nb_agents,))
#         arr_k = np.array(self.graph.df_attributes['K'])
#         if arr_pop is None:
#             arr_pop = self.count_pop_per_vertex(position_attribute='territory')
#         arr_pop = np.array(arr_pop, dtype=np.int32)
#         arr_stop = _temp_random_walk_on_sphere_exit_random_walk_based_on_k(arr_selected_agents, rand, prob_settlement,
#                                                                            alpha, arr_pos_selected_agents, arr_k,
#                                                                            arr_pop)
#         # print('stopping agents:', arr_stop)
#         self.df_population['is_on_random_walk'] = self.df_population['is_on_random_walk'] & ~arr_stop
#         # print(self.df_population[['territory', 'col_id']])
#         # print(self.df_population['territory'])
#         self.df_population['territory'] = self.df_population['territory'] + \
#                                           arr_stop * (arr_pos_selected_agents - self.df_population['territory'])
#         # print(' ')
#         # print(self.df_population['territory'])
#         # print('---------')
#         # time.sleep(2.)
#
#         self.df_population['position'] = self.df_population['position'] + \
#                                           arr_stop * (arr_pos_selected_agents - self.df_population['position'])
#         return arr_stop

