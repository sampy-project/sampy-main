from ...agent.base import BaseAgingAgent
from ...agent.mortality import (NaturalMortalityOrmMethodology, 
                                OffspringDependantOnParents, 
                                MortalityKillTooOld,
                                KillPercentagePop)
from ...agent.reproduction import (OffspringCreationWithCustomProb,
                                   FindMateMonogamous)
from ...agent.movement import TerritorialMovementWithoutResistance, SwitchMovementCriterion
from ...utils.decorators import sampy_class
from ...pandas_xs.pandas_xs import DataFrameXS
from .jit_compiled_functions import (movement_directional_dispersion_with_varying_nb_of_steps, 
                                     movement_directional_dispersion_with_varying_nb_of_steps_return_path,
                                     update_id_if_needed,
                                     update_rabies_if_needed_with_vacc,
                                     update_rabies_if_needed)

import numpy as np

class FoxesDirectionalMovementWithResistance:
    """
    This building block defines two new types of movements meant to represent
    two different types of dispersal movements, that we call Type 1 and Type 2 
    dispersal (to our knowledge, this terminology is not used in an external 
    source and is, therefore, specific to the current documentation).

    Requires:
        1) graph structure:
            The agents using this building-block HAVE TO live on an oriented
            hexagonal grid, or any graph derived / apparented to it. A non
            exhaustive list of such graphs is :
                - graph/builting_graph/OrientedHexagonalLattice (basic graph)
                - addons/GIS_interface/geographic_grid/HexGrid (for interfacing with GIS softwares)
                - addons/ORM_related_addons/graph_from_ORM_xml/GraphFromORMxml (if your graph comes from 
                    historical ORM-based XML format)

        2) resistance:
            All movements defined in this Building Block interact with two expected 
            attributes of the graphs vertices, respectively called in_res and out_res.

            - in_res: a float between 0 and 1 per vertex. Models how difficult it
                is for an agent to enter the vertex while dispersing. When entering
                a vertex, a bernoulli test with the vertex in_res as parameter is done.
                if successful, the movement fails and the agent does not enter the vertex.
            - out_res: a float between 0 and 1 per vertex. Models how difficult it
                is for an agent to exit the vertex while dispersing. Same math logic 
                as in_res.

    Type 1 movement: 
        Dipersion with a constant direction. I.e. the agents pick a direction when its 
        dispersal starts and it tries to keep this direction constant as much as 
        possible. The agent may change direction only when encountering a border
        or when failing a resistance test.
        IMPORTANT: this type of dispersal allows for some "wigling" using user defined
            parameters. I.e. the agents will not, unless configured to do so, go in
            perfectly straight lines.

    Type 2 movement:
        Dispersion with a varying direction. This movement is a discretized 
        analog of usual "correlated random walks" used in ecological modelling.
        I.e. agents tends to preserve their initial direction, but it may 
        change during movement depending on user provided parameters.
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

        self.df_population['territory'] = None
        self.df_population['position'] = None
        self.add_attribute('direction', -1)
        self.add_attribute('reinitialize_direction', True)

# base mouvement
    def _directional_dispersion_from_arr_nb_steps(self, 
                                                 arr_nb_steps,
                                                 arr_directional_prob,                             
                                                 condition=None,
                                                 territory_attribute='territory',
                                                 position_attribute='position',
                                                 direction_attribute='direction',
                                                 in_res_attribute='in_res',
                                                 out_res_attribute='out_res',
                                                 return_path=False,
                                                 update_dir=False,
                                                 nb_retry_res=2,
                                                 stop_mov=False):
        """
        Internal method for movement modelling.

        Used to modelize dispersion of agents. Each selected agent will perform a 
        user defined number of steps on the graph. Both position and territory are updated.

        :param arr_nb_steps: 1D array of int, giving the permissible number of steps.
        :param arr_directional_prob: array of float of shape (6,). Controls how the direction may change between
            two consecutive steps. If we fix the following notation for this array [a, b, c, d, e, f], then:
                - a is the probability to make a 180 turn;
                - b is the probability to turn 120 degrees to the left;
                - c is the probability to turn 60 degree to the left
                - d is the probability to continue in the same direction as the previous step;
                - e is the probability to turn 60 degree to the right;
                - f is the probability to turn 120 degree to the right.
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move. If left at None, all the agents move.
        :param territory_attribute: optional, string, default 'territory'.
        :param position_attribute: optional, string, default 'position'.
        :param direction_attribute: optional, string, default 'direction'.
        :param in_res_attribute: optional, string, default 'in_res'.
        :param out_res_attribute: optional, string, default 'out_res'.
        :param return_path: optional, bool, default 'False'. If 'True' the method returns a Numba dictionary where the keys are agents IDs 
                            and the values are their path as list of vertices IDs. If 'False' return none.
        :param update_dir: optional, bool, default 'False'. If 'False' the direction is conserved as long as the steps are successful. If 'True' 
                           the direction is updated after each step.
        :param nb_retry_res: optional, int, default '2'. Number of retry after an agent is blocked by resistance.
        :param stop_mov: optional, bool, default False. If True, the movement of the agent is stopped if it fails to perform a step
                         after nb_retry_res attempt (this would be cause by the resistance)
        """

        if self.df_population.nb_rows == 0:
            return

        arr_prob = arr_directional_prob.astype('float64')
        arr_cumul_directional_prob = np.cumsum(arr_prob)
        arr_cumul_directional_prob = arr_cumul_directional_prob / arr_cumul_directional_prob[-1]
        arr_cumul_directional_prob[-1] = 1.


        if condition is not None:
            arr_nb_steps = arr_nb_steps * condition



        max_nb_rand = 2 * (nb_retry_res + 1) * arr_nb_steps.sum()
        arr_rand = np.random.uniform(0, 1, max_nb_rand)
        arr_rand_res = np.random.uniform(0, 1, arr_rand.shape)
        if return_path:
            return movement_directional_dispersion_with_varying_nb_of_steps_return_path(self.df_population[territory_attribute], 
                                                                    self.df_population[position_attribute], 
                                                                    arr_nb_steps, arr_cumul_directional_prob,
                                                                    self.graph.connections, self.graph.weights,
                                                                    arr_rand, arr_rand_res,
                                                                    self.df_population[direction_attribute], self.df_population['col_id'], update_dir, 
                                                                    self.graph.df_attributes[in_res_attribute], 
                                                                    self.graph.df_attributes[out_res_attribute],
                                                                    nb_retry_res, stop_mov)

        movement_directional_dispersion_with_varying_nb_of_steps(self.df_population[territory_attribute], 
                                                                    self.df_population[position_attribute], 
                                                                    arr_nb_steps, arr_cumul_directional_prob,
                                                                    self.graph.connections, self.graph.weights,
                                                                    arr_rand, arr_rand_res,
                                                                    self.df_population[direction_attribute],
                                                                    self.graph.df_attributes[in_res_attribute], 
                                                                    self.graph.df_attributes[out_res_attribute],
                                                                    nb_retry_res, stop_mov)

       


     # TYPE 1
    def dispersion_with_constant_direction(self, 
                                            arr_nb_steps,
                                            arr_directional_prob, 
                                            arr_nb_timestep_inf=None,
                                            arr_prob_nb_timestep_inf=None,                            
                                            condition=None,
                                            territory_attribute='territory',
                                            position_attribute='position',
                                            direction_attribute='direction',
                                            in_res_attribute='in_res',
                                            out_res_attribute='out_res',
                                            return_path=False, 
                                            disease = None,
                                            vaccine = None,
                                            is_border_attribute='is_border',
                                            prob_change_id_attribute='prob_change_id',
                                            prob_change_rabies_attribute='prob_change_rabies',
                                            nb_retry_res=2,
                                            stop_mov=False):
        """
        Corresponds to the "type 1" movement described in the class's dosctring. Both position 
        and territory are updated.

        :param arr_nb_steps: 1D array of int, giving the permissible number of steps.
        :param arr_directional_prob: array of float of shape (6,). Controls how the direction of a step may deviate
            from the prefered direction. A deviation from this parameters DO NOT change the prefered
            direction (only a border or resistance can change it). If we fix the following notation 
            for this array [a, b, c, d, e, f], then:
                - a is the probability to make a 180 turn;
                - b is the probability to turn 120 degrees to the left;
                - c is the probability to turn 60 degree to the left
                - d is the probability to continue in the same direction as the previous step;
                - e is the probability to turn 60 degree to the right;
                - f is the probability to turn 120 degree to the right.
        :param arr_nb_timestep_inf: optional, 1D array of int, default None. If not None, work in tandem with arr_prob_nb_timestep_inf.
        :param arr_prob_nb_timestep_inf: optional, 1D array of float, default None. If not None, arr_prob_nb_timestep_inf[i] 
                                        is the probability for an agent to stay infected but not contagious for 
                                        arr_nb_timestep_inf[i] time-steps.
        :param condition: optional, array of booleans, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move. If left at None, all the agents move.
        :param territory_attribute: optional, string, default 'territory'.
        :param position_attribute: optional, string, default 'position'.
        :param direction_attribute: optional, string, default 'direction'.
        :param in_res_attribute: optional, string, default 'in_res'.
        :param out_res_attribute: optional, string, default 'out_res'.
        :param return_path: optional, bool, default 'False'. If 'True' the method returns a Numba dictionary where the keys are agents IDs 
                            and the values are their path as list of vertices IDs. If 'False' return none.
        :param disease: optional, disease object, default 'None'.
        :param vaccine: optional, vaccination object, default 'None'
        :param is_border_attribute: optional, string, default 'is_border'. Name of the graph attribute telling which vertices are on the border.
        :param prob_change_id_attribute: optional, string, default 'prob_change_id'. Name of the graph attribute telling 
                                        the probability of changing ID when the agent cross a border vertex. 
        :param prob_change_rabies_attribute: optional, string, default 'prob_change_rabies'. Name of the graph attribute telling 
                                        the probability of changing rabies status when the agent cross a border vertex.
        :param nb_retry_res: optional, int, default '2'. Number of retry after an agent is blocked by resistance.
        :param stop_mov: optional, bool, default False. If True, the movement of the agent is stopped if it fails to perform a step
                         after nb_retry_res attempt (this would be cause by the resistance)
        """

        if self.df_population.nb_rows == 0:
            return

        dict_path = self._directional_dispersion_from_arr_nb_steps(arr_nb_steps, arr_directional_prob, condition=condition,
                                                      territory_attribute=territory_attribute,
                                                      position_attribute=position_attribute,
                                                      direction_attribute=direction_attribute,
                                                      in_res_attribute=in_res_attribute,
                                                      out_res_attribute=out_res_attribute,
                                                      return_path=True,
                                                      update_dir=False,
                                                      nb_retry_res=nb_retry_res,
                                                      stop_mov=stop_mov)
        
        new_id, arr_has_change_id, arr_last_border_cell = update_id_if_needed(self.df_population['col_id'], self.graph.df_attributes[is_border_attribute],
                            dict_path, self.counter_id, self.graph.df_attributes[prob_change_id_attribute],
                            np.random.uniform(0, 1, self.df_population.nb_rows))
        self.counter_id = new_id

        if disease is not None:
            disease_name = disease.disease_name
            arr_inf = self.df_population['inf_' + disease_name]
            arr_inf_cnt = self.df_population['cnt_inf_' + disease_name]
            arr_con = self.df_population['con_' + disease_name]
            arr_con_cnt = self.df_population['cnt_con_' + disease_name]
            arr_imm = self.df_population['imm_' + disease_name]

            sample_nb_timestep_inf = np.random.choice(arr_nb_timestep_inf, arr_has_change_id.sum(), 
                                                        p=arr_prob_nb_timestep_inf)
            sample_nb_timestep_inf -= 1
            sample_nb_timestep_inf[sample_nb_timestep_inf <= 0] = 1

            if vaccine is not None:
                if vaccine.disease.disease_name != disease_name:
                    raise ValueError(f"The vaccine is not associated with the disease {disease_name}.")
                
                arr_vacc = self.df_population['vaccinated_' + disease_name]
                arr_vacc_cnt = self.df_population['cnt_vaccinated_' + disease_name]

                update_rabies_if_needed_with_vacc(arr_has_change_id, arr_last_border_cell, 
                                                  self.graph.df_attributes[prob_change_rabies_attribute],
                                                  arr_inf, arr_inf_cnt, arr_con, arr_con_cnt,
                                                  arr_imm, np.random.uniform(0, 1, arr_has_change_id.sum()),
                                                  sample_nb_timestep_inf, arr_vacc, arr_vacc_cnt)
            else:
                
                update_rabies_if_needed(arr_has_change_id, arr_last_border_cell,
                                        self.graph.df_attributes[prob_change_rabies_attribute],
                                        arr_inf, arr_inf_cnt, arr_con, arr_con_cnt,
                                        arr_imm, np.random.uniform(0, 1, arr_has_change_id.sum()),
                                        sample_nb_timestep_inf)
        if return_path:
            return dict_path

     # TYPE 2
    def dispersion_with_varying_direction(self, 
                                            arr_nb_steps,
                                            arr_directional_prob,                             
                                            condition=None,
                                            territory_attribute='territory',
                                            position_attribute='position',
                                            direction_attribute='direction',
                                            in_res_attribute='in_res',
                                            out_res_attribute='out_res',
                                            nb_retry_res=2,
                                            return_path=False,
                                            stop_mov=False):
        """
        Corresponds to the "type 2" movement described in the class's dosctring. Both position 
        and territory are updated.

        :param arr_nb_steps: 1D array of int, giving the permissible number of steps. 
        :param arr_directional_prob: array of float of shape (6,). Controls how the direction may change between
            two consecutive steps. If we fix the following notation for this array [a, b, c, d, e, f], then:
                - a is the probability to make a 180 turn;
                - b is the probability to turn 120 degrees to the left;
                - c is the probability to turn 60 degree to the left
                - d is the probability to continue in the same direction as the previous step;
                - e is the probability to turn 60 degree to the right;
                - f is the probability to turn 120 degree to the right.
        :param condition: optional, array of booleans, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move. If left at None, all the agents move.
        :param territory_attribute: optional, string, default 'territory'.
        :param position_attribute: optional, string, default 'position'.
        :param direction_attribute: optional, string, default 'direction'.
        :param in_res_attribute: optional, string, default 'in_res'.
        :param out_res_attribute: optional, string, default 'out_res'.
        :param nb_retry_res: optional, int, default '2'. Number of retry after an agent is blocked by resistance.
        :param return_path: optional, bool, default 'False'. If 'True' the method returns a Numba dictionary where the keys are agents IDs 
                            and the values are their path as list of vertices IDs. If 'False' return none.
        :param stop_mov: optional, bool, default False. If True, the movement of the agent is stopped if it fails to perform a step
                         after nb_retry_res attempt (this would be cause by the resistance)
        """

        if self.df_population.nb_rows == 0:
            return
        if return_path:
            return self._directional_dispersion_from_arr_nb_steps(arr_nb_steps, 
                                                       arr_directional_prob, condition=condition,
                                                      territory_attribute=territory_attribute,
                                                      position_attribute=position_attribute,
                                                      direction_attribute=direction_attribute,
                                                      in_res_attribute=in_res_attribute,
                                                      out_res_attribute=out_res_attribute,
                                                      return_path=True,
                                                      update_dir=True,
                                                      nb_retry_res=nb_retry_res,
                                                      stop_mov=stop_mov)
        else:
            self._directional_dispersion_from_arr_nb_steps(arr_nb_steps, 
                                                       arr_directional_prob, condition=condition,
                                                      territory_attribute=territory_attribute,
                                                      position_attribute=position_attribute,
                                                      direction_attribute=direction_attribute,
                                                      in_res_attribute=in_res_attribute,
                                                      out_res_attribute=out_res_attribute,
                                                      return_path=False,
                                                      update_dir=True,
                                                      nb_retry_res=nb_retry_res,
                                                      stop_mov=stop_mov)
        
    def choose_nb_steps_per_agent_from_custom_prob(self, arr_nb_step_possible, arr_prob_nb_step_possible):
        """
        Get an array of steps per agent from user defined probabilities.

        :param arr_nb_step_possible: 1D array of int. work in tandem with arr_prob_nb_step_possible.
        :param arr_prob_nb_step_possible: 1D array of float. arr_prob_nb_step_possible[i] is the probability for 
                                        an agent to make arr_nb_step_possible[i] steps.

        :returns: array of int of shape (nb_agents,)
        """
        return np.random.choice(arr_nb_step_possible, self.df_population.nb_rows, p=arr_prob_nb_step_possible)
    
    def choose_nb_steps_per_agent_from_gamma_distrib(self, mu, scale, max_nb_step=None, condition=None):
        """
        Get an array of steps per agent from a user-defined gamma distribution. The sampling comes from
        the function numpy.random.gamma (doc can be found at 
        https://numpy.org/doc/2.3/reference/random/generated/numpy.random.gamma.html )

        WARNING: getting samples from a gamma distribution is computation intensive. 
            Consequently, it is recommanded to use the kwarg condition when picking a number of steps.

        :param mu: float
        :param scale: float
        :param max_nb_step: optional int, default 'None'. If not 'None', no value of nb steps can be above.
        :param condition: optional, arr of bool, default None

        :returns: array of int of shape (nb_agents,)
        """
        shape = mu/scale
        if condition is None:
            arr_nb_steps = np.floor(np.random.gamma(shape, scale, self.df_population.nb_rows)).astype(int)
        else:
            arr_nb_steps = np.full(self.df_population.nb_rows, 0)
            arr_nb_steps[condition] = np.floor(np.random.gamma(shape, scale, condition.sum())).astype(int)

        if max_nb_step is not None:
            arr_nb_steps[arr_nb_steps >= max_nb_step] = max_nb_step
        return arr_nb_steps


@sampy_class
class Foxes(BaseAgingAgent,
            MortalityKillTooOld,
            NaturalMortalityOrmMethodology,
            OffspringDependantOnParents,
            FindMateMonogamous,
            OffspringCreationWithCustomProb,
            TerritorialMovementWithoutResistance,
            SwitchMovementCriterion,
            KillPercentagePop,    
            FoxesDirectionalMovementWithResistance):
    """
    This class represents a population of Foxes (either Red or Arctic) for regional modelling.
    The main functionalities of this class relates to dispersion movement as defined in the above 
    building block FoxesDirectionalMovementWithResistance.
    
    NOTE: for very large scale (e.g. Pan-Arctic), consider using the class BasicMammalWithSCRW in agent/builtin_agent.py
    """
    def __init__(self, **kwargs):
        pass
