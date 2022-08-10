import numpy as np
from graph_dynamic.utils.jit_compiled_functions import (conditional_count_return_full_array,
                                                        conditional_count_return_full_array_two_species)


class TransmissionByContactWithSameGraph:
    def __init__(self):
        super().__init__()

    def intra_specie_contact_contagion(self, host, contact_rate, position_attribute='position',
                                       conditional=False, arr_condition=None,
                                       lambda_poisson=None,
                                       custom_transition_arr_nb_stimstep=None, custom_transition_arr_cum_prob=None,
                                       return_arr_new_infected=False):
        """
        Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
        count the number of contagious agents N_c, then each non immuned agent on the cell X has a probability of
        1 - (1 - contact_rate) ** N_c to become infected.

        :param host: host of the disease
        :param contact_rate: Float value. used to determine the probability of becoming infected
        :param position_attribute: optional, string, default 'position'. Name of the agent attribute used as position.
        :param conditional: optional, boolean, default to False. Set to True if some agent are protected from infection
                            because of any custom reason.
        :param arr_condition: Used only if conditional is set to True. Array of boolean sur that the i-th value is
                              True if and only if the i-th agent (i.e. the agent at the line i of df_population) can be
                              infected. All the agent having their corresponding value to False are protected from
                              infection.
        :param lambda_poisson: todo
        :param custom_transition_arr_nb_stimstep: todo
        :param custom_transition_arr_cum_prob: todo
        :param return_arr_new_infected: todo
        """
        col_pos = np.array(host.df_population[position_attribute], dtype=np.int32)
        col_con = np.array(host.df_population['con_' + self.disease_name], dtype=bool)
        nb_vertex = host.graph.connections.shape[0]

        # return the array counting the number of contagious agents
        count_con = conditional_count_return_full_array(nb_vertex, col_pos, col_con)

        # make the array of newly infected individuals. Note that for the moment the computation is not optimized,
        # and the random contamination is computed for EVERY agent, and then we exclude the not susceptible ones.
        new_infected = np.random.uniform(0, 1, (count_con.shape[0],)) < 1 - (1 - contact_rate) ** count_con

        # exclusion of the immuned and already infected agents
        new_infected = new_infected & ~(host.df_population['inf_' + self.disease_name]) \
                                    & ~(host.df_population['con_' + self.disease_name]) \
                                    & ~(host.df_population['imm_' + self.disease_name])

        if conditional:
            # only keep the ones for which the condition is satisfied
            host.df_population['inf_' + self.disease_name] = (host.df_population[
                                                             'inf_' + self.disease_name] | new_infected) & arr_condition
        else:
            host.df_population['inf_' + self.disease_name] = host.df_population['inf_' + self.disease_name] | new_infected

        if hasattr(self, '_initialize_on_infection_custom_prob'):
            if custom_transition_arr_nb_stimstep is None or custom_transition_arr_cum_prob is None:
                raise ValueError("A value for the key-word arguments custom_transition_arr_nb_stimstep and custom_" +
                                 "transition_arr_cum_prob should be given")
            self._initialize_on_infection_custom_prob(host,
                                                      new_infected,
                                                      custom_transition_arr_nb_stimstep,
                                                      custom_transition_arr_cum_prob)

        if return_arr_new_infected:
            return new_infected

    def inter_specie_contact_contagion(self, contact_rate, source_host=None, target_host=None,
                                       position_attribute='position',
                                       conditional=False, arr_condition=None,
                                       lambda_poisson=None,
                                       custom_transition_arr_nb_stimstep=None, custom_transition_arr_cum_prob=None,
                                       return_arr_new_infected=False):
        """
        Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
        count the number of contagious agents N_c, then each non immuned agent on the cell X has a probability of
        1 - (1 - contact_rate) ** N_c to become infected.

        :param contact_rate: Float value. used to determine the probability of becoming infected
        :param source_host: host of the disease that will contaminate the target host
        :param target_host: host of the disease that will be contaminated by the source host
        :param position_attribute: optional, string, default 'position'. Name of the agent attribute used as position.
        :param conditional: optional, boolean, default to False. Set to True if some agent are protected from infection
                            because of any custom reason.
        :param arr_condition: Used only if conditional is set to True. Array of boolean sur that the i-th value is
                              True if and only if the i-th agent (i.e. the agent at the line i of df_population) can be
                              infected. All the agent having their corresponding value to False are protected from
                              infection.
        :param lambda_poisson: todo
        :param custom_transition_arr_nb_stimstep: todo
        :param custom_transition_arr_cum_prob: todo
        :param return_arr_new_infected: todo
        """
        if (source_host is None) or (target_host is None):
            raise ValueError("A value for the key-word arguments source_hoste and target_host should be given")

        col_pos_target = np.array(target_host.df_population[position_attribute], dtype=np.int32)
        col_pos_source = np.array(source_host.df_population[position_attribute], dtype=np.int32)
        col_con_source = np.array(source_host.df_population['con_' + self.disease_name], dtype=bool)
        nb_vertex = source_host.graph.connections.shape[0]

        # return the array counting the number of contagious agents
        count_con = conditional_count_return_full_array_two_species(nb_vertex, col_pos_source,
                                                                    col_con_source, col_pos_target)

        # make the array of newly infected individuals. Note that for the moment the computation is not optimized,
        # and the random contamination is computed for EVERY agent, and then we exclude the not susceptible ones.
        new_infected = np.random.uniform(0, 1, (count_con.shape[0],)) < 1 - (1 - contact_rate) ** count_con

        # exclusion of the immuned and already infected agents
        new_infected = new_infected & ~(target_host.df_population['inf_' + self.disease_name]) \
                                    & ~(target_host.df_population['con_' + self.disease_name]) \
                                    & ~(target_host.df_population['imm_' + self.disease_name])

        if conditional:
            # only keep the ones for which the condition is satisfied
            target_host.df_population['inf_' + self.disease_name] = (target_host.df_population[
                                                        'inf_' + self.disease_name] | new_infected) & arr_condition
        else:
            target_host.df_population['inf_' + self.disease_name] = target_host.df_population[
                                                                 'inf_' + self.disease_name] | new_infected

        if hasattr(self, '_initialize_on_infection_custom_prob'):
            if custom_transition_arr_nb_stimstep is None or custom_transition_arr_cum_prob is None:
                raise ValueError("A value for the key-word arguments custom_transition_arr_nb_stimstep and custom_" +
                                 "transition_arr_cum_prob should be given")
            self._initialize_on_infection_custom_prob(target_host,
                                                      new_infected,
                                                      custom_transition_arr_nb_stimstep,
                                                      custom_transition_arr_cum_prob)

        if return_arr_new_infected:
            return new_infected
