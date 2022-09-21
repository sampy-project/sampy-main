import numpy as np
from .jit_compiled_functions import *
from ...utils.errors_shortcut import (check_input_array,
                                      check_col_exists_good_type)


class TransmissionByContact:
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_contact_contagion(self, contact_rate, position_attribute='position', condition=None,
                                       return_arr_new_infected=True):
        if self.host.df_population.nb_rows == 0:
            return

        check_col_exists_good_type(self.host.df_population, position_attribute, 'position_attribute',
                                   prefix_dtype='int', reject_none=True)
        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.host.df_population.nb_rows,))

    def contact_contagion(self, contact_rate, position_attribute='position',
                          condition=None,
                          return_arr_new_infected=True):
        """
        Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
        count the number of contagious agents N_c, then each non immuned agent on the vertex X has a probability of

                            1 - (1 - contact_rate) ** N_c

        to become infected.

        :param contact_rate: Float value. used to determine the probability of becoming infected
        :param position_attribute: optional, string, default 'position'. Name of the agent attribute used as position.
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can be
                          infected and transmit disease. All the agent having their corresponding value to False are
                          protected from infection and cannot transmit the disease.
        :param return_arr_new_infected: optional, bool, default True
        """
        col_pos = self.host.df_population[position_attribute]
        col_con = self.host.df_population['con_' + self.disease_name]
        if condition is not None:
            col_con = col_con & condition
        nb_vertex = self.host.graph.connections.shape[0]

        # return the array counting the number of contagious agents
        count_con = conditional_count_return_full_array(nb_vertex, col_pos, col_con)

        # make the array of newly infected individuals. Note that for the moment the computation is not optimized,
        # and the random contamination is computed for EVERY agent, and then we exclude the not susceptible ones.
        new_infected = np.random.uniform(0, 1, (count_con.shape[0],)) < 1 - (1 - contact_rate) ** count_con

        # exclusion of the immuned and already infected agents
        new_infected = new_infected & ~(self.host.df_population['inf_' + self.disease_name]) \
                                    & ~(self.host.df_population['con_' + self.disease_name]) \
                                    & ~(self.host.df_population['imm_' + self.disease_name])

        if condition is not None:
            new_infected = new_infected & condition

        self.host.df_population['inf_' + self.disease_name] = self.host.df_population['inf_' + self.disease_name] | new_infected

        if return_arr_new_infected:
            return new_infected


class TransmissionByContactWithContactTracing:
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_contact_contagion(self, contact_rate, position_attribute='position', condition=None,
                                       return_arr_new_infected=True):
        if self.host.df_population.nb_rows == 0:
            return

        check_col_exists_good_type(self.host.df_population, position_attribute, 'position_attribute',
                                   prefix_dtype='int', reject_none=True)
        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.host.df_population.nb_rows,))

    def contact_contagion(self, contact_rate, position_attribute='position', id_attribute='col_id',
                          condition=None,
                          return_arr_new_infected=True):
        """
        Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
        count the number of contagious agents N_c, then each non immuned agent on the cell X has a probability of

                            1 - (1 - contact_rate) ** N_c

        to become infected. This class will assign to each newly contaminated agent an individual who
        contaminated it. This comes with an extra computational cost.

        :param contact_rate: Float value. used to determine the probability of becoming infected
        :param position_attribute: optional, string, default 'position'. Name of the agent attribute used as position.
        :param id_attribute: optional, string, default 'col_id'.
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can be
                          infected. All the agent having their corresponding value to False are protected from
                          infection.
        :param return_arr_new_infected: optional, bool, default True
        """
        col_pos = self.host.df_population[position_attribute]
        col_con = self.host.df_population['con_' + self.disease_name]
        nb_vertex = self.host.graph.connections.shape[0]

        # return the array counting the number of contagious agents
        count_con = conditional_count_return_full_array(nb_vertex, col_pos, col_con)

        # make the array of newly infected individuals. Note that for the moment the computation is not optimized,
        # and the random contamination is computed for EVERY agent, and then we exclude the not susceptible ones.
        new_infected = np.random.uniform(0, 1, (count_con.shape[0],)) < 1 - (1 - contact_rate) ** count_con

        # exclusion of the immuned and already infected agents
        new_infected = new_infected & ~(self.host.df_population['inf_' + self.disease_name]) \
                                    & ~(self.host.df_population['con_' + self.disease_name]) \
                                    & ~(self.host.df_population['imm_' + self.disease_name])

        if condition is not None:
            new_infected = new_infected & condition

        self.host.df_population['inf_' + self.disease_name] = self.host.df_population['inf_' + self.disease_name] | new_infected

        # here we assign to each newly infected agent a 'source of disease'
        rand_con = np.random.uniform(0, 1, new_infected.sum())
        col_id, col_contact = transmission_contact_contagion_contact_tracing(self.host.df_population[id_attribute],
                                                                             col_pos,
                                                                             col_con,
                                                                             rand_con,
                                                                             new_infected,
                                                                             nb_vertex)

        if return_arr_new_infected:
            return new_infected, col_id, col_contact
        else:
            return col_id, col_contact

    def create_csv_contact_list(self, list_arr_id, list_arr_contact, list_timestep=None):
        list_non_zero_arr = [(len(arr.shape) > 0 and arr.shape[0] > 0) for arr in list_arr_id]
        final_list_arr_id = [arr for i, arr in enumerate(list_arr_id) if list_non_zero_arr[i]]
        final_list_arr_contact = [arr for i, arr in enumerate(list_arr_contact) if list_non_zero_arr[i]]
