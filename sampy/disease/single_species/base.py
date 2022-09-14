import numpy as np
from .jit_compiled_functions import *
from ...utils.errors_shortcut import check_col_exists_good_type


class BaseSingleSpeciesDisease:
    def __init__(self, disease_name=None, host=None, **kwargs):
        # check values have been given
        if host is None:
            raise ValueError("No host given for the disease. Use the kwarg 'host'.")
        if disease_name is None:
            raise ValueError("No name given to the disease. Use the kwarg 'disease_name'.")

        self.host = host
        self.disease_name = disease_name

        self.host.df_population['inf_' + disease_name] = False
        self.host.df_population['con_' + disease_name] = False
        self.host.df_population['imm_' + disease_name] = False

        if hasattr(host, 'dict_default_val'):
            self.host.dict_default_val['inf_' + disease_name] = False
            self.host.dict_default_val['con_' + disease_name] = False
            self.host.dict_default_val['imm_' + disease_name] = False

        if not hasattr(self, 'list_disease_status'):
            self.set_disease_status = {'inf', 'con', 'imm'}
        else:
            self.set_disease_status.update(['inf', 'con', 'imm'])

        self.on_ticker = []

    def tick(self):
        """
        execute in order all the methods whose name are in the list 'on_ticker'. Those methods should not accept
        any arguments.
        """
        for method in self.on_ticker:
            getattr(self, method)()

    def _sampy_debug_count_nb_status_per_vertex(self, target_status, position_attribute='position'):
        if self.host.df_population.nb_rows == 0:
            return
        check_col_exists_good_type(self.host.df_population, position_attribute, 'attribute_position',
                                   prefix_dtype='int', reject_none=True)
        check_col_exists_good_type(self.host.df_population, target_status + '_' + self.disease_name,
                                   'target_status', prefix_dtype='bool', reject_none=True)

    def count_nb_status_per_vertex(self, target_status, position_attribute='position'):
        """
        Count the number of agent having the targeted status in each vertex. The status can either be 'inf', 'con' and
        'imm', which respectively corresponds to infected, contagious and immunized agents.

        :param target_status: string in ['inf', 'con', 'imm'].
        :param position_attribute: optional, string.

        :return: array counting the number of agent having the target status in each vertex
        """
        if self.host.df_population.nb_rows == 0:
            return np.full((self.host.graph.number_vertices,), 0, dtype=np.int32)
        return base_conditional_count_nb_agent_per_vertex(self.host.df_population[target_status + '_' +
                                                                                  self.disease_name],
                                                          self.host.df_population[position_attribute],
                                                          self.host.graph.weights.shape[0])

    def contaminate_vertices(self, list_vertices, level, return_arr_newly_contaminated=True,
                             condition=None, position_attribute='position'):
        """
        Contaminate the vertices given in the list 'list_vertices' with the disease. Each agent on the vertex have a
        probability of 'level' to be contaminated.

        :param list_vertices: list of vertices ID to be contaminated.
        :param level: float, probability for agent on the vertices to be contaminated
        :param return_arr_newly_contaminated: optional, boolean, default True. If True, the method returns an array
                                              telling which agents were contaminated.
        :param condition: optional, array of bool, default None. If not None, say which agents are susceptible to be
                          contaminated.
        :param position_attribute: optional, string, default 'position'. Agent attribute to be used to define
                                   their position.
        :return: if return_arr_newly_contaminated is set to True, returns a 1D array of bool. Otherwise, returns
                 None.
        """
        for i, vertex_id in enumerate(list_vertices):
            if i == 0:
                arr_new_infected = (self.host.df_population[position_attribute] ==
                                    self.host.graph.dict_cell_id_to_ind[vertex_id])
                continue
            arr_new_infected = arr_new_infected | (self.host.df_population[position_attribute] ==
                                                   self.host.graph.dict_cell_id_to_ind[vertex_id])
        arr_new_infected = arr_new_infected & ~(self.host.df_population['inf_' + self.disease_name] |
                                                self.host.df_population['con_' + self.disease_name] |
                                                self.host.df_population['imm_' + self.disease_name])
        if condition is not None:
            arr_new_infected = arr_new_infected & condition
        rand = np.random.uniform(0, 1, (arr_new_infected.sum(),))
        base_contaminate_vertices(arr_new_infected, rand, level)
        self.host.df_population['inf_' + self.disease_name] = \
            self.host.df_population['inf_' + self.disease_name] | arr_new_infected
        if return_arr_newly_contaminated:
            return arr_new_infected
