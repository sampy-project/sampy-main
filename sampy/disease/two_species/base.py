import numpy as np
from .jit_compiled_functions import base_contaminate_vertices


class BaseTwoSpeciesDisease:
    """Base class for two species disease. This building block expects the following kwargs:

    :param disease_name: mandatory kwargs. String.
    :param host1: mandatory kwargs. Population object of the first host.
    :param host2: mandatory kwargs. Population object of the second host.
    """
    def __init__(self, disease_name='', host1=None, host2=None, **kwargs):
        # check values have been given
        if not host1:
            raise ValueError('No first host given for the disease. Use Kwarg host1.')
        if not host2:
            raise ValueError('No second host given for the disease. Use Kwarg host2.')
        if not disease_name:
            raise ValueError('No name given to the disease. Use Kwarg disease_name.')

        self.host1 = host1
        self.host2 = host2
        self.disease_name = disease_name

        self.host1.df_population['inf_' + disease_name] = False
        self.host1.df_population['con_' + disease_name] = False
        self.host1.df_population['imm_' + disease_name] = False

        if hasattr(host1, 'dict_default_val'):
            self.host1.dict_default_val['inf_' + disease_name] = False
            self.host1.dict_default_val['con_' + disease_name] = False
            self.host1.dict_default_val['imm_' + disease_name] = False

        self.host2.df_population['inf_' + disease_name] = False
        self.host2.df_population['con_' + disease_name] = False
        self.host2.df_population['imm_' + disease_name] = False

        if hasattr(host2, 'dict_default_val'):
            self.host2.dict_default_val['inf_' + disease_name] = False
            self.host2.dict_default_val['con_' + disease_name] = False
            self.host2.dict_default_val['imm_' + disease_name] = False

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

    def contaminate_vertices(self, list_vertices, level, host, return_arr_newly_contaminated=True,
                             condition=None, position_attribute='position'):
        """
        Contaminate the vertices given in the list 'list_vertices' with the disease. Each agent on the vertex have a
        probability of 'level' to be contaminated.

        :param list_vertices: list of vertices ID to be contaminated.
        :param level: float, probability for agent on the vertices to be contaminated
        :param host: int, either 1 or 2. if 1, host 1 will be contaminated, if 2, 2 will be.
        :param return_arr_newly_contaminated: optional, boolean, default True. If True, the method returns an array
                                              telling which agents were contaminated.
        :param condition: optional, array of bool, default None. If not None, say which agents are susceptible to be
                          contaminated.
        :param position_attribute: optional, string, default 'position'. Agent attribute to be used to define
                                   their position.
        :return: if return_arr_newly_contaminated is set to True, returns a 1D array of bool. Otherwise, returns
                 None.
        """
        if host == 1:
            host = self.host1
        elif host == 2:
            host = self.host2
        else:
            raise ValueError('Host arguments should be an integer, and either 1 or 2.')

        for i, vertex_id in enumerate(list_vertices):
            if i == 0:
                arr_new_infected = (host.df_population[position_attribute] ==
                                    host.graph.dict_cell_id_to_ind[vertex_id])
                continue
            arr_new_infected = arr_new_infected | (host.df_population[position_attribute] ==
                                                   host.graph.dict_cell_id_to_ind[vertex_id])
        arr_new_infected = arr_new_infected & ~(host.df_population['inf_' + self.disease_name] |
                                                host.df_population['con_' + self.disease_name] |
                                                host.df_population['imm_' + self.disease_name])
        if condition is not None:
            arr_new_infected = arr_new_infected & condition
        rand = np.random.uniform(0, 1, (arr_new_infected.sum(),))
        base_contaminate_vertices(arr_new_infected, rand, level)
        host.df_population['inf_' + self.disease_name] = host.df_population['inf_' + self.disease_name] | \
                                                         arr_new_infected
        if return_arr_newly_contaminated:
            return arr_new_infected
