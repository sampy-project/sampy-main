import numpy as np
from graph_dynamic.utils.jit_compiled_functions import (falsify_when_condition,
                                                        zero_when_condition,
                                                        fill_poisson_counter,
                                                        fill_custom_transition_counter)


class TransitionCustomProbPermanentImmunity:
    def __init__(self):
        self.host1.df_population['cnt_inf_' + self.disease_name] = 0
        self.host1.df_population['cnt_con_' + self.disease_name] = 0

        self.host2.df_population['cnt_inf_' + self.disease_name] = 0
        self.host2.df_population['cnt_con_' + self.disease_name] = 0

        super().__init__()

    def _initialize_on_infection_custom_prob(self, host, arr_new_infected, arr_nb_timestep, arr_cum_prob):
        """
        todo
        :param host: which host should be initialize
        :param arr_new_infected: array of boolean saying which individuals are newly infected (and only those !)
        :param arr_nb_timestep:
        :param arr_cum_prob:
        :return:
        """
        nb_new_infected = arr_new_infected.sum()
        arr_cnt = np.array(host.df_population['cnt_inf_' + self.disease_name], dtype=np.int32)
        uniform_sample = np.random.uniform(0, 1, (nb_new_infected,))
        host.df_population['cnt_inf_' + self.disease_name] = fill_custom_transition_counter(arr_cnt,
                                                                                            arr_new_infected,
                                                                                            arr_nb_timestep,
                                                                                            arr_cum_prob,
                                                                                            uniform_sample)

    def decrement_counter(self):
        """
        todo
        """
        self.host1.df_population['cnt_inf_' + self.disease_name] -= self.host.df_population['inf_' + self.disease_name]
        self.host1.df_population['cnt_con_' + self.disease_name] -= self.host.df_population['con_' + self.disease_name]

        self.host2.df_population['cnt_inf_' + self.disease_name] -= self.host.df_population['inf_' + self.disease_name]
        self.host2.df_population['cnt_con_' + self.disease_name] -= self.host.df_population['con_' + self.disease_name]

    def transition_with_decreasing_counter(self, host, initial_state, target_state, proba_death=None,
                                           arr_nb_timestep=None, arr_prob_nb_timestep=None):
        """
        Deterministically perform the transition from an initial_state to a target_state.
        :param initial_state: string, in ['inf', 'con', 'imm']
        :param target_state: string, in ['con', 'imm', 'death']
        :param proba_death: float, optional. Give probability of death if target_state=death. Default value 1.0
        :param arr_nb_timestep: todo
        :param arr_prob_nb_timestep: todo
        """
        # bool array of all individuals that will make transition
        susceptible = np.array(host.df_population['cnt_' + initial_state + '_' + self.disease_name] == 0, dtype=bool)
        susceptible = susceptible & np.array(host.df_population[initial_state + '_' + self.disease_name], dtype=bool)
        # in case of death
        if target_state == 'death':
            # there might be a probability of dying of the disease, which is taken into account now
            if proba_death is not None:
                susceptible = susceptible & \
                              (np.random.uniform(0, 1, (host.df_population.shape[0],)) < proba_death)
            # killing
            host.df_population = host.df_population[~susceptible].copy()

        # all the rest corresponds to transition between stages of the disease
        else:
            host.df_population[target_state + '_' + self.disease_name] = susceptible | host.df_population[
                                                                                target_state + '_' + self.disease_name]
            if target_state == 'imm':
                bool_inf = np.array(host.df_population['inf_' + self.disease_name], dtype=bool)
                host.df_population['inf_' + self.disease_name] = falsify_when_condition(bool_inf, susceptible)

                bool_con = np.array(host.df_population['con_' + self.disease_name], dtype=bool)
                host.df_population['con_' + self.disease_name] = falsify_when_condition(bool_con, susceptible)

            else:
                if arr_nb_timestep is None or arr_prob_nb_timestep is None:
                    raise ValueError("No probability arrays given for disease transition")

                bool_ini_state = np.array(host.df_population[initial_state + '_' + self.disease_name], dtype=bool)
                host.df_population[initial_state + '_' + self.disease_name] = falsify_when_condition(bool_ini_state,
                                                                                                     susceptible)
                nb_new_infected = susceptible.sum()
                arr_cnt = np.array(host.df_population['cnt_' + target_state + '_' + self.disease_name], dtype=np.int32)
                uniform_sample = np.random.uniform(0, 1, (nb_new_infected,))
                host.df_population['cnt_' + target_state + '_' + self.disease_name] = \
                    fill_custom_transition_counter(arr_cnt,
                                                   susceptible,
                                                   arr_nb_timestep,
                                                   arr_prob_nb_timestep,
                                                   uniform_sample)

