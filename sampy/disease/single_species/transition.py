import numpy as np
from .jit_compiled_functions import *
from ...utils.errors_shortcut import (check_input_array,
                                      check_col_exists_good_type)


class TransitionCustomProbPermanentImmunity:
    """
    This class introduce transitions between disease states based on probabilities given by the user.

    WARNING: be aware that the time each agent spend in each status of the disease is kept in memory using counters.
             Those counters HAVE TO be initialized for newly infected individuals throught the use of the method
             'initialize_counters_of_newly_infected'. This is a current problem of Sampy that the user has to
             explicitly call a method that should be automatically called in the background. This issue will be
             adressed in the future once a satisfactory design solution has been found (here, satisfactory means
             'that doesn't create too much special cases that developers working on Sampy have to keep in mind').
    """
    def __init__(self, **kwargs):
        self.host.df_population['cnt_inf_' + self.disease_name] = 0
        self.host.dict_default_val['cnt_inf_' + self.disease_name] = 0

        self.host.df_population['cnt_con_' + self.disease_name] = 0
        self.host.dict_default_val['cnt_con_' + self.disease_name] = 0

        self.on_ticker.append('decrement_counter')

    def _sampy_debug_initialize_counters_of_newly_infected(self, arr_new_infected, arr_nb_timestep, arr_prob):
        if self.host.df_population.nb_rows == 0:
            return

        check_input_array(arr_new_infected, 'arr_new_infected', 'bool',
                          shape=(self.host.df_population.nb_rows,))
        check_input_array(arr_nb_timestep, 'arr_nb_timestep', 'int', nb_dim=1)
        check_input_array(arr_prob, 'arr_prob', 'float', nb_dim=1)

        if arr_prob.shape != arr_nb_timestep.shape:
            raise ValueError("Arguments 'arr_nb_timestep' and 'arr_prob' have different shapes.")

    def initialize_counters_of_newly_infected(self, arr_new_infected, arr_nb_timestep, arr_prob):
        """
        Method that HAS TO be called each time new individuals get infected

        :param arr_new_infected: 1d array of bool, saying which agent are newly infected and should have their
                                 'infectious status counter' initialized.
        :param arr_nb_timestep: 1d array of int.
        :param arr_prob: 1d array of non negative floats, will be normalized to sum to 1.

        """
        if self.host.df_population.nb_rows == 0:
            return
        prob = arr_prob.astype('float64')
        prob = prob/prob.sum()
        arr_cnt = np.random.choice(arr_nb_timestep, arr_new_infected.sum(), p=prob)

        transition_initialize_counters_of_newly_infected(arr_new_infected,
                                                         self.host.df_population['cnt_inf_' + self.disease_name],
                                                         arr_cnt)

    def decrement_counter(self):
        """
        Reduce by one the counters of the agents in a given disease state. Note that this method will only decrease
        positive counters, so that if a negative counter was to appear, which shouldn't, this should be caused by
        something else.
        """
        if self.host.df_population.nb_rows == 0:
            return
        self.host.df_population['cnt_inf_' + self.disease_name] -= self.host.df_population['inf_' + self.disease_name] & \
                                                        (self.host.df_population['cnt_inf_' + self.disease_name] > 0)
        self.host.df_population['cnt_con_' + self.disease_name] -= self.host.df_population['con_' + self.disease_name] & \
                                                        (self.host.df_population['cnt_con_' + self.disease_name] > 0)

    def _sampy_debug_transition_between_states(self, initial_state, target_state, condition=None, proba_death=1.,
                                               arr_nb_timestep=None, arr_prob_nb_timestep=None,
                                               return_transition_count=False, position_attribute='position'):
        if self.host.df_population.nb_rows == 0:
            return

        if initial_state not in self.set_disease_status:
            raise ValueError("Initial state is not in " + str(self.set_disease_status) + ".")
        if target_state not in self.set_disease_status and target_state != 'death':
            raise ValueError("Initial state is not in " + str(self.set_disease_status | {'death'}) + ".")

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.host.df_population.nb_rows,))

        if arr_nb_timestep is None and arr_prob_nb_timestep is not None:
            raise ValueError("'arr_nb_timestep' is None while 'arr_prob_nb_timestep' is not.")

        if arr_nb_timestep is not None and arr_prob_nb_timestep is None:
            raise ValueError("'arr_prob_nb_timestep' is None while 'arr_nb_timestep' is not.")

        if arr_nb_timestep is not None:
            check_input_array(arr_nb_timestep, 'arr_nb_timestep', 'int', nb_dim=1)
            check_input_array(arr_prob_nb_timestep, 'arr_prob_nb_timestep', 'float', nb_dim=1)
            if arr_nb_timestep.shape != arr_prob_nb_timestep.shape:
                raise ValueError("Arguments 'arr_nb_timestep' and 'arr_prob_nb_timestep' have different shapes.")

        check_col_exists_good_type(self.host.df_population, position_attribute, 'position_attribute',
                                   prefix_dtype='int', reject_none=True)

    def transition_between_states(self, initial_state, target_state, condition=None, proba_death=1.,
                                  arr_nb_timestep=None, arr_prob_nb_timestep=None, return_transition_count=False,
                                  position_attribute='position'):
        """
        Performs the transition from an initial_state to a target_state of the disease, where 'death' is a possible
        target_state. When performing the transition, each agent for which 'initial_state' is True and the associated
        counter is 0 makes the transition (except if the target state is death proba_death is smaller than 1.).

        WARNING: note that an agent can only be in a SINGLE state. That is, an agent cannot be simultaneously 'infected'
                 and 'contagious'. So if the user wants, for instance, to count all agents carrying the disease, then
                 both 'inf' and 'con' states have to be considered.

        :param initial_state: string, in ['inf', 'con', 'imm']
        :param target_state: string, in ['con', 'imm', 'death']
        :param proba_death: optional, float, default 1.0. Probability of death if target_state=death.
        :param arr_nb_timestep: optional, 1d array of int, default None.
        :param arr_prob_nb_timestep: optional, 1d array of float, default None.
        :param return_transition_count: optional, bool, default False. If True, returns a 1D array of integer counting
                                        how many agents did the transition per cell.
        :param position_attribute: optional, string, default 'position'. Name of the position attribute used for counting
                                   if 'return_transition_count' is set to True.

        :returns: if return_transition_count is True, returns a 1d array of int. Else, returns None.
        """
        if self.host.df_population.nb_rows == 0:
            return

        # bool array of all individuals that will make transition
        susceptible = self.host.df_population['cnt_' + initial_state + '_' + self.disease_name] == 0
        susceptible = susceptible & self.host.df_population[initial_state + '_' + self.disease_name]
        if condition is not None:
            susceptible = susceptible & condition

        count_arr = None

        # in case of death
        if target_state == 'death':
            # there might be a probability of dying of the disease, which is taken into account now
            if proba_death < 1.:
                susceptible = susceptible & \
                              (np.random.uniform(0, 1, (self.host.df_population.shape[0],)) < proba_death)

            if return_transition_count:
                count_arr = transition_conditional_count_nb_agent_per_vertex(susceptible,
                                                                             self.host.df_population[
                                                                                 position_attribute],
                                                                             self.host.graph.weights.shape[0])

            # killing
            self.host.df_population = self.host.df_population[~susceptible]

        # all the rest corresponds to transition between stages of the disease
        else:
            if return_transition_count:
                count_arr = transition_conditional_count_nb_agent_per_vertex(susceptible,
                                                                             self.host.df_population[
                                                                                 position_attribute],
                                                                             self.host.graph.weights.shape[0])

            self.host.df_population[target_state + '_' + self.disease_name] = susceptible | self.host.df_population[
                                                                                target_state + '_' + self.disease_name]
            if target_state == 'imm':
                transition_falsify_when_condition(self.host.df_population['inf_' + self.disease_name], susceptible)
                transition_falsify_when_condition(self.host.df_population['con_' + self.disease_name], susceptible)

            else:
                transition_falsify_when_condition(self.host.df_population[initial_state + '_' + self.disease_name],
                                                  susceptible)

                prob = arr_prob_nb_timestep.astype('float64')
                prob = prob / prob.sum()
                arr_cnt = np.random.choice(arr_nb_timestep, susceptible.sum(), p=prob)

                transition_initialize_counters_of_newly_infected(susceptible,
                                                                 self.host.df_population[
                                                                     'cnt_' + target_state + '_' + self.disease_name],
                                                                 arr_cnt)
        return count_arr


class TransitionCustomProbFiniteImmunity:
    """
    This class introduce transitions between disease states based on probabilities given by the user. Immunity is
    not permanent.

    WARNING: be aware that the time each agent spend in each status of the disease is kept in memory using counters.
             Those counters HAVE TO be initialized for newly infected individuals throught the use of the method
             'initialize_counters_of_newly_infected'. This is a current problem of Sampy that the user has to
             explicitly call a method that should be automatically called in the background. This issue will be
             adressed in the future once a satisfactory design solution has been found (here, satisfactory means
             'that doesn't create too much special cases that developers working on Sampy have to keep in mind').
    """
    def __init__(self, **kwargs):
        self.host.df_population['cnt_inf_' + self.disease_name] = 0
        self.host.dict_default_val['cnt_inf_' + self.disease_name] = 0

        self.host.df_population['cnt_con_' + self.disease_name] = 0
        self.host.dict_default_val['cnt_con_' + self.disease_name] = 0

        self.host.df_population['cnt_imm_' + self.disease_name] = 0
        self.host.dict_default_val['cnt_imm_' + self.disease_name] = 0

    def _sampy_debug_initialize_counters_of_newly_infected(self, arr_new_infected, arr_nb_timestep, arr_prob):
        if self.host.df_population.nb_rows == 0:
            return

        check_input_array(arr_new_infected, 'arr_new_infected', 'bool',
                          shape=(self.host.df_population.nb_rows,))
        check_input_array(arr_nb_timestep, 'arr_nb_timestep', 'int', nb_dim=1)
        check_input_array(arr_prob, 'arr_prob', 'float', nb_dim=1)

        if arr_prob.shape != arr_nb_timestep:
            raise ValueError("Arguments 'arr_nb_timestep' and 'arr_prob' have different shapes.")

    def initialize_counters_of_newly_infected(self, arr_new_infected, arr_nb_timestep, arr_prob):
        """
        Method that HAS TO be called each time new individuals get infected

        :param arr_new_infected: 1d array of bool, saying which agent are newly infected and should have their
                                 'infectious status counter' initialized.
        :param arr_nb_timestep: 1d array of int.
        :param arr_prob: 1d array of non negative floats, will be normalized to sum to 1.

        """
        if self.host.df_population.nb_rows == 0:
            return
        prob = arr_prob.astype('float64')
        prob = prob/prob.sum()
        arr_cnt = np.random.choice(arr_nb_timestep, arr_new_infected.sum(), p=prob)

        transition_initialize_counters_of_newly_infected(arr_new_infected,
                                                         self.host.df_population['cnt_inf_' + self.disease_name],
                                                         arr_cnt)

    def decrement_counter(self):
        """
        Reduce by one the counters of the agents in a given disease state. Note that this method will only decrease
        positive counters, so that if a negative counter was to appear, which shouldn't, this should be caused by
        something else.
        """
        if self.host.df_population.nb_rows == 0:
            return
        self.host.df_population['cnt_inf_' + self.disease_name] -= self.host.df_population['inf_' + self.disease_name] & \
                                                        (self.host.df_population['cnt_inf_' + self.disease_name] > 0)
        self.host.df_population['cnt_con_' + self.disease_name] -= self.host.df_population['con_' + self.disease_name] & \
                                                        (self.host.df_population['cnt_con_' + self.disease_name] > 0)
        self.host.df_population['cnt_imm_' + self.disease_name] -= self.host.df_population['imm_' + self.disease_name] & \
                                                                   (self.host.df_population[
                                                                        'cnt_imm_' + self.disease_name] > 0)

    def _sampy_debug_transition_between_states(self, initial_state, target_state, condition=None, proba_death=1.,
                                               arr_nb_timestep=None, arr_prob_nb_timestep=None,
                                               return_transition_count=False, position_attribute='position'):
        if self.host.df_population.nb_rows == 0:
            return

        if initial_state not in self.set_disease_status:
            raise ValueError("Initial state is not in " + str(self.set_disease_status) + ".")
        if target_state not in self.set_disease_status and target_state != 'death':
            raise ValueError("Initial state is not in " + str(self.set_disease_status | {'death'}) + ".")

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.host.df_population.nb_rows,))

        if arr_nb_timestep is None and arr_prob_nb_timestep is not None:
            raise ValueError("'arr_nb_timestep' is None while 'arr_prob_nb_timestep' is not.")

        if arr_nb_timestep is not None and arr_prob_nb_timestep is None:
            raise ValueError("'arr_prob_nb_timestep' is None while 'arr_nb_timestep' is not.")

        if arr_nb_timestep is not None:
            check_input_array(arr_nb_timestep, 'arr_nb_timestep', 'int', nb_dim=1)
            check_input_array(arr_prob_nb_timestep, 'arr_prob_nb_timestep', 'float', nb_dim=1)
            if arr_nb_timestep.shape != arr_prob_nb_timestep.shape:
                raise ValueError("Arguments 'arr_nb_timestep' and 'arr_prob_nb_timestep' have different shapes.")

        check_col_exists_good_type(self.host.df_population, position_attribute, 'position_attribute',
                                   prefix_dtype='int', reject_none=True)

    def transition_between_states(self, initial_state, target_state, condition=None, proba_death=1.,
                                  arr_nb_timestep=None, arr_prob_nb_timestep=None, return_transition_count=False,
                                  position_attribute='position'):
        """
        Performs the transition from an initial_state to a target_state of the disease, where 'death' is a possible
        target_state. When performing the transition, each agent for which 'initial_state' is True and the associated
        counter is 0 makes the transition (except if the target state is death and proba_death is smaller than 1.).

        WARNING: note that an agent can only be in a SINGLE state. That is, an agent cannot be simultaneously 'infected'
                 and 'contagious'. So if the user wants, for instance, to count all agents carrying the disease, then
                 both 'inf' and 'con' states have to be considered.

        :param initial_state: string, in ['inf', 'con', 'imm']
        :param target_state: string, in ['con', 'imm', 'death']
        :param proba_death: optional, float, default 1.0. Probability of death if target_state=death.
        :param arr_nb_timestep: optional, 1d array of int, default None.
        :param arr_prob_nb_timestep: optional, 1d array of float, default None.
        :param return_transition_count: optional, bool, default False. If True, returns a 1D array of integer counting
                                        how many agents did the transition per cell.
        :param position_attribute: optional, string, default 'position'. Name of the position attribute used for counting
                                   if 'return_transition_count' is set to True.

        :returns: if return_transition_count is True, returns a 1d array of int. Else, returns None.
        """
        if self.host.df_population.nb_rows == 0:
            return

        # bool array of all individuals that will make transition
        susceptible = self.host.df_population['cnt_' + initial_state + '_' + self.disease_name] == 0
        susceptible = susceptible & self.host.df_population[initial_state + '_' + self.disease_name]
        if condition is not None:
            susceptible = susceptible & condition

        count_arr = None

        # in case of death
        if target_state == 'death':
            # there might be a probability of dying of the disease, which is taken into account now
            if proba_death < 1.:
                susceptible = susceptible & \
                              (np.random.uniform(0, 1, (self.host.df_population.shape[0],)) < proba_death)

            if return_transition_count:
                count_arr = transition_conditional_count_nb_agent_per_vertex(susceptible,
                                                                             self.host.df_population[
                                                                                 position_attribute],
                                                                             self.host.graph.weights.shape[0])

            # killing
            self.host.df_population = self.host.df_population[~susceptible]

        # all the rest corresponds to transition between stages of the disease
        else:
            if return_transition_count:
                count_arr = transition_conditional_count_nb_agent_per_vertex(susceptible,
                                                                             self.host.df_population[
                                                                                 position_attribute],
                                                                             self.host.graph.weights.shape[0])

            self.host.df_population[target_state + '_' + self.disease_name] = susceptible | self.host.df_population[
                                                                                target_state + '_' + self.disease_name]

            transition_falsify_when_condition(self.host.df_population[initial_state + '_' + self.disease_name],
                                              susceptible)

            prob = arr_prob_nb_timestep.astype('float64')
            prob = prob / prob.sum()
            arr_cnt = np.random.choice(arr_nb_timestep, susceptible.sum(), p=prob)

            transition_initialize_counters_of_newly_infected(susceptible,
                                                             self.host.df_population[
                                                                 'cnt_' + target_state + '_' + self.disease_name],
                                                             arr_cnt)
        return count_arr
