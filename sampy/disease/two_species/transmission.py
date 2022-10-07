import numpy as np
from .jit_compiled_functions import (transmission_conditional_count,
                                     transmission_count_needed_samples,
                                     transmission_disease_propagation,
                                     transmission_disease_propagation_return_new_inf,
                                     transmission_disease_propagation_return_type_inf)
from ...utils.errors_shortcut import (check_input_array,
                                      check_col_exists_good_type)


class ContactTransmissionSameGraph:
    """
    The disease is spread by direct contact between the agents. Both populations are assumed to live on the same graph.
    """
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_contact_contagion(self, contact_rate_matrix, position_attribute_host1='position',
                                       position_attribute_host2='position', condition_host1=None, condition_host2=None,
                                       return_arr_new_infected=False, return_type_transmission=False):
        if not isinstance(contact_rate_matrix, np.ndarray):
            raise ValueError("contact_rate_matrix argument should be a ndarray of shape (2, 2).")
        if self.host1.df_population.nb_rows > 0:
            check_col_exists_good_type(self.host1.df_population, position_attribute_host1, 'position_attribute_host1',
                                       prefix_dtype='int', reject_none=True)
            if condition_host1 is not None:
                check_input_array(condition_host1, "condition_host1", "bool", shape=(self.host1.df_population.nb_rows,))
        if self.host2.df_population.nb_rows > 0:
            check_col_exists_good_type(self.host2.df_population, position_attribute_host2, 'position_attribute_host2',
                                       prefix_dtype='int', reject_none=True)
            if condition_host2 is not None:
                check_input_array(condition_host2, "condition_host2", "bool", shape=(self.host2.df_population.nb_rows,))

    def contact_contagion(self, contact_rate_matrix, position_attribute_host1='position',
                          position_attribute_host2='position', condition_host1=None, condition_host2=None,
                          return_arr_new_infected=False, return_type_transmission=False):
        """
        Propagate the disease by direct contact using the following methodology.

        We denote by N_{v, host_i} the number of contagious agents of the species host_i on the vertex v of the graph.
        Then, for any susceptible agent of the species host_j living on the vertex v, the probability for this agent to
        be contaminated by an infected agent of the species host_i is given by:

                            1 - (1 - contact_rate_{i,j}) ** N_{c, host_i}

        In practice, for any susceptible agent, we perform two independent tests (one for each host species) and if any
        of those test is a success, the agent is contaminated.

        :param contact_rate_matrix: 2D array of floats of shape (2, 2). Here, contact_rate_matrix[0][0] is the
                                    probability of contact contagion from host1 to host1, contact_rate_matrix[0][1] is
                                    the probability of contact contagion from host1 to host2, etc...
        :param position_attribute_host1: optional, string, default 'position'. Name of the agent attribute used as
                                         position for host1.
        :param position_attribute_host2: optional, string, default 'position'. Name of the agent attribute used as
                                         position for host2.
        :param condition_host1: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent of host1 (i.e. the agent at the line i of df_population)
                          can be infected and transmit disease. All the agent having their corresponding value to False
                          are protected from infection and cannot transmit the disease.
        :param condition_host2: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent of host2 (i.e. the agent at the line i of df_population)
                          can be infected and transmit disease. All the agent having their corresponding value to False
                          are protected from infection and cannot transmit the disease.
        :param return_arr_new_infected: optional, boolean, default False. If True, the method returns two arrays telling
                                        which agent got contaminated in each host species.
        :param return_type_transmission: optional, boolean, default False.

        :return: Depending on the values of the parameters 'return_arr_new_infected' and 'return_type_transmission',
                 the return value will either be None (both are False) or a dictionnary whose key-values are:
                    - 'arr_new_infected_host1', 'arr_new_infected_host2' if return_arr_new_infected is True, and the
                      values are 1D arrays of bool telling which agents got infected for each host;
                    - 'arr_type_transmission_host1', 'arr_type_transmission_host2' if return_type_transmission is True,
                      and the values are 1D arrays of non-negative integers such that the integer at line i is 0 if the
                      i-th agent has not been contaminated, 1 if it has been contaminated by a member of its own species
                      2 if it has been contaminated by a member of the other species, 3 if it has been contaminated by
                      agents from both its species and the other.
        """
        if self.host1.df_population.nb_rows > 0:
            pos_host1 = self.host1.df_population[position_attribute_host1]
            contagious_host1 = self.host1.df_population['con_' + self.disease_name]
            susceptible_host1 = ~(self.host1.df_population['inf_' + self.disease_name] |
                                  self.host1.df_population['con_' + self.disease_name] |
                                  self.host1.df_population['imm_' + self.disease_name])
            if condition_host1 is not None:
                contagious_host1 = contagious_host1 & condition_host1
                susceptible_host1 = susceptible_host1 & condition_host1
            count_con_host1 = transmission_conditional_count(self.host1.graph.number_vertices, pos_host1,
                                                             contagious_host1)
        else:
            if self.host2.df_population.nb_rows == 0:
                return
            # not the best hack, but the case where one of the two species died off should not be very common,
            # todo: change this for a properly optimized branch instead of this dirty hack
            pos_host1 = np.array([0])
            count_con_host1 = np.full((self.host1.graph.number_vertices,), 0, dtype=int)
            susceptible_host1 = np.array([False])

        if self.host2.df_population.nb_rows > 0:
            pos_host2 = self.host2.df_population[position_attribute_host2]
            contagious_host2 = self.host2.df_population['con_' + self.disease_name]
            susceptible_host2 = ~(self.host2.df_population['inf_' + self.disease_name] |
                                  self.host2.df_population['con_' + self.disease_name] |
                                  self.host2.df_population['imm_' + self.disease_name])
            if condition_host2 is not None:
                contagious_host2 = contagious_host2 & condition_host2
                susceptible_host1 = susceptible_host1 & condition_host1
            count_con_host2 = transmission_conditional_count(self.host2.graph.number_vertices, pos_host2,
                                                             contagious_host2)
        else:
            # the case of both species having died out has already been treated.
            # todo: see above, same problem of dirty hack
            pos_host2 = np.array([0])
            count_con_host2 = np.full((self.host2.graph.number_vertices,), 0, dtype=int)
            susceptible_host2 = np.array([False])

        # we now take care of the contaminations
        nb_samples_host1 = transmission_count_needed_samples(susceptible_host1, pos_host1, count_con_host1,
                                                             count_con_host2)
        nb_samples_host2 = transmission_count_needed_samples(susceptible_host2, pos_host2, count_con_host1,
                                                             count_con_host2)
        random_numbers = np.random.uniform(0, 1, (nb_samples_host1 + nb_samples_host2,))

        rv = None
        if return_arr_new_infected and not return_type_transmission:
            rv = {}
            new_inf_host1, new_inf_host2 = transmission_disease_propagation_return_new_inf(
                                                                   susceptible_host1, susceptible_host2,
                                                                   position_attribute_host1, position_attribute_host2,
                                                                   self.host1.df_population['inf_' + self.disease_name],
                                                                   self.host2.df_population['inf_' + self.disease_name],
                                                                   count_con_host1, count_con_host2,
                                                                   random_numbers, contact_rate_matrix)
            rv['arr_new_infected_host1'] = new_inf_host1
            rv['arr_new_infected_host2'] = new_inf_host2
            return rv
        if return_type_transmission:
            rv = {}
            type_inf_host1, type_inf_host2 = transmission_disease_propagation_return_type_inf(
                                                                   susceptible_host1, susceptible_host2,
                                                                   position_attribute_host1, position_attribute_host2,
                                                                   self.host1.df_population['inf_' + self.disease_name],
                                                                   self.host2.df_population['inf_' + self.disease_name],
                                                                   count_con_host1, count_con_host2,
                                                                   random_numbers, contact_rate_matrix)
            rv['arr_type_transmission_host1'] = type_inf_host1
            rv['arr_type_transmission_host2'] = type_inf_host2
            if return_arr_new_infected:
                rv['arr_new_infected_host1'] = type_inf_host1 > 0
                rv['arr_new_infected_host2'] = type_inf_host2 > 0
            return rv

        # case where nothing is returned
        transmission_disease_propagation(susceptible_host1, susceptible_host2,
                                         position_attribute_host1, position_attribute_host2,
                                         self.host1.df_population['inf_' + self.disease_name],
                                         self.host2.df_population['inf_' + self.disease_name],
                                         count_con_host1, count_con_host2,
                                         random_numbers, contact_rate_matrix)
        return rv


    # def intra_specie_contact_contagion(self, host, contact_rate, position_attribute='position',
    #                                    conditional=False, arr_condition=None,
    #                                    lambda_poisson=None,
    #                                    custom_transition_arr_nb_stimstep=None, custom_transition_arr_cum_prob=None,
    #                                    return_arr_new_infected=False):
    #     """
    #     Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
    #     count the number of contagious agents N_c, then each non immuned agent on the cell X has a probability of
    #     1 - (1 - contact_rate) ** N_c to become infected.
    #
    #     :param host: host of the disease
    #     :param contact_rate: Float value. used to determine the probability of becoming infected
    #     :param position_attribute: optional, string, default 'position'. Name of the agent attribute used as position.
    #     :param conditional: optional, boolean, default to False. Set to True if some agent are protected from infection
    #                         because of any custom reason.
    #     :param arr_condition: Used only if conditional is set to True. Array of boolean sur that the i-th value is
    #                           True if and only if the i-th agent (i.e. the agent at the line i of df_population) can be
    #                           infected. All the agent having their corresponding value to False are protected from
    #                           infection.
    #     :param lambda_poisson: todo
    #     :param custom_transition_arr_nb_stimstep: todo
    #     :param custom_transition_arr_cum_prob: todo
    #     :param return_arr_new_infected: todo
    #     """
    #     col_pos = np.array(host.df_population[position_attribute], dtype=np.int32)
    #     col_con = np.array(host.df_population['con_' + self.disease_name], dtype=bool)
    #     nb_vertex = host.graph.connections.shape[0]
    #
    #     # return the array counting the number of contagious agents
    #     count_con = conditional_count_return_full_array(nb_vertex, col_pos, col_con)
    #
    #     # make the array of newly infected individuals. Note that for the moment the computation is not optimized,
    #     # and the random contamination is computed for EVERY agent, and then we exclude the not susceptible ones.
    #     new_infected = np.random.uniform(0, 1, (count_con.shape[0],)) < 1 - (1 - contact_rate) ** count_con
    #
    #     # exclusion of the immuned and already infected agents
    #     new_infected = new_infected & ~(host.df_population['inf_' + self.disease_name]) \
    #                                 & ~(host.df_population['con_' + self.disease_name]) \
    #                                 & ~(host.df_population['imm_' + self.disease_name])
    #
    #     if conditional:
    #         # only keep the ones for which the condition is satisfied
    #         host.df_population['inf_' + self.disease_name] = (host.df_population[
    #                                                          'inf_' + self.disease_name] | new_infected) & arr_condition
    #     else:
    #         host.df_population['inf_' + self.disease_name] = host.df_population['inf_' + self.disease_name] | new_infected
    #
    #     if hasattr(self, '_initialize_on_infection_custom_prob'):
    #         if custom_transition_arr_nb_stimstep is None or custom_transition_arr_cum_prob is None:
    #             raise ValueError("A value for the key-word arguments custom_transition_arr_nb_stimstep and custom_" +
    #                              "transition_arr_cum_prob should be given")
    #         self._initialize_on_infection_custom_prob(host,
    #                                                   new_infected,
    #                                                   custom_transition_arr_nb_stimstep,
    #                                                   custom_transition_arr_cum_prob)
    #
    #     if return_arr_new_infected:
    #         return new_infected
    #
    # def inter_specie_contact_contagion(self, contact_rate, source_host=None, target_host=None,
    #                                    position_attribute='position',
    #                                    conditional=False, arr_condition=None,
    #                                    lambda_poisson=None,
    #                                    custom_transition_arr_nb_stimstep=None, custom_transition_arr_cum_prob=None,
    #                                    return_arr_new_infected=False):
    #     """
    #     Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
    #     count the number of contagious agents N_c, then each non immuned agent on the cell X has a probability of
    #     1 - (1 - contact_rate) ** N_c to become infected.
    #
    #     :param contact_rate: Float value. used to determine the probability of becoming infected
    #     :param source_host: host of the disease that will contaminate the target host
    #     :param target_host: host of the disease that will be contaminated by the source host
    #     :param position_attribute: optional, string, default 'position'. Name of the agent attribute used as position.
    #     :param conditional: optional, boolean, default to False. Set to True if some agent are protected from infection
    #                         because of any custom reason.
    #     :param arr_condition: Used only if conditional is set to True. Array of boolean sur that the i-th value is
    #                           True if and only if the i-th agent (i.e. the agent at the line i of df_population) can be
    #                           infected. All the agent having their corresponding value to False are protected from
    #                           infection.
    #     :param lambda_poisson: todo
    #     :param custom_transition_arr_nb_stimstep: todo
    #     :param custom_transition_arr_cum_prob: todo
    #     :param return_arr_new_infected: todo
    #     """
    #     if (source_host is None) or (target_host is None):
    #         raise ValueError("A value for the key-word arguments source_hoste and target_host should be given")
    #
    #     col_pos_target = np.array(target_host.df_population[position_attribute], dtype=np.int32)
    #     col_pos_source = np.array(source_host.df_population[position_attribute], dtype=np.int32)
    #     col_con_source = np.array(source_host.df_population['con_' + self.disease_name], dtype=bool)
    #     nb_vertex = source_host.graph.connections.shape[0]
    #
    #     # return the array counting the number of contagious agents
    #     count_con = conditional_count_return_full_array_two_species(nb_vertex, col_pos_source,
    #                                                                 col_con_source, col_pos_target)
    #
    #     # make the array of newly infected individuals. Note that for the moment the computation is not optimized,
    #     # and the random contamination is computed for EVERY agent, and then we exclude the not susceptible ones.
    #     new_infected = np.random.uniform(0, 1, (count_con.shape[0],)) < 1 - (1 - contact_rate) ** count_con
    #
    #     # exclusion of the immuned and already infected agents
    #     new_infected = new_infected & ~(target_host.df_population['inf_' + self.disease_name]) \
    #                                 & ~(target_host.df_population['con_' + self.disease_name]) \
    #                                 & ~(target_host.df_population['imm_' + self.disease_name])
    #
    #     if conditional:
    #         # only keep the ones for which the condition is satisfied
    #         target_host.df_population['inf_' + self.disease_name] = (target_host.df_population[
    #                                                     'inf_' + self.disease_name] | new_infected) & arr_condition
    #     else:
    #         target_host.df_population['inf_' + self.disease_name] = target_host.df_population[
    #                                                              'inf_' + self.disease_name] | new_infected
    #
    #     if hasattr(self, '_initialize_on_infection_custom_prob'):
    #         if custom_transition_arr_nb_stimstep is None or custom_transition_arr_cum_prob is None:
    #             raise ValueError("A value for the key-word arguments custom_transition_arr_nb_stimstep and custom_" +
    #                              "transition_arr_cum_prob should be given")
    #         self._initialize_on_infection_custom_prob(target_host,
    #                                                   new_infected,
    #                                                   custom_transition_arr_nb_stimstep,
    #                                                   custom_transition_arr_cum_prob)
    #
    #     if return_arr_new_infected:
    #         return new_infected
