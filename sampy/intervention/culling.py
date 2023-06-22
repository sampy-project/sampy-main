import numpy as np
from .jit_compiled_functions import culling_apply_culling_from_array_condition


class BaseCullingSingleSpecies:
    def __init__(self, species=None, **kwargs):
        if species is None:
            raise ValueError(
                "No agent object provided for the culling. Should be provided using kwarg 'species'.")
        self.species = species


class CullingSingleSpecies:
    def __init__(self, **kwargs):
        pass

    def apply_culling_from_array(self, array_culling_level, condition=None, position_attribute='position'):
        """
        Kill proportion of agents based on the 1D array 'array_culling_level'. array_culling_level[i] is the
        probability for an agent on the vertex of index i to be killed.

        By default, all agent can be killed. Use kwarg 'condition' to refine the culling.

        :param array_culling_level: 1D array of float
        :param condition: optional, 1D array of bool, default None.
        :param position_attribute: optionnal, string, default 'position'
        """
        if condition is None:
            condition = np.full((self.species.df_population.nb_rows,), True, dtype=np.bool_)

        rand = np.random.uniform(0, 1, (condition.sum(),))

        survive_culling = culling_apply_culling_from_array_condition(array_culling_level,
                                                                     self.species.df_population[position_attribute],
                                                                     rand, condition)

        self.species.df_population = self.species.df_population[survive_culling]

    def apply_culling_from_dict(self, graph, dict_vertex_id_to_level, condition=None, position_attribute='position'):
        """
        Same as apply_culling_from_array, but the 1D array is replaced by a dictionary whose keys are vertices ID and
        values is the culling level on each cell.

        :param graph: graph object on which the culling is applied
        :param dict_vertex_id_to_level: dictionnary-like object with culling level
        :param condition: optional, 1D array of bool, default None.
        :param position_attribute: optional, string, default 'position'.
        """
        array_cul_level = np.full((graph.number_vertices,), 0., dtype=float)
        for id_vertex, level in dict_vertex_id_to_level.items():
            array_cul_level[graph.dict_cell_id_to_ind[id_vertex]] = level
        self.apply_culling_from_array(array_cul_level, condition=condition, position_attribute=position_attribute)
