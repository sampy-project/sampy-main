import numpy as np
from graph_dynamic.utils.jit_compiled_functions import conditional_count_nb_agent_per_vertex


class BaseTwoSpeciesDisease:
    def __init__(self, disease_name='', host1=None, host2=None):
        """
        todo
        :param host:
        """
        # check values have been given
        if not host1:
            raise ValueError('No first host given for the disease')
        if not host2:
            raise ValueError('No second host given for the disease')
        if not disease_name:
            raise ValueError('No name given to the disease')
        self.host1 = host1
        self.host2 = host2
        self.disease_name = disease_name

        self.host1.df_population['inf_' + disease_name] = False
        self.host1.df_population['con_' + disease_name] = False
        self.host1.df_population['imm_' + disease_name] = False

        self.host2.df_population['inf_' + disease_name] = False
        self.host2.df_population['con_' + disease_name] = False
        self.host2.df_population['imm_' + disease_name] = False

        super().__init__()

    def count_nb_status_per_vertex(self, host, target_status, attribute_position='position'):
        """
        Count the number of agent having the targeted status in each vertex.
        :param host:
        :param target_status: string in ['inf', 'con', 'imm']
        :param attribute_position: optional, string.
        :return: array counting the number of agent having the target status in each vertex
        """
        nb_vertex = host.graph.weights.shape[0]
        pos = np.array(host.df_population[attribute_position], dtype=np.int32)
        condition = np.array(host.df_population[target_status + '_' + self.disease_name], dtype=bool)
        return conditional_count_nb_agent_per_vertex(condition, pos, nb_vertex)
