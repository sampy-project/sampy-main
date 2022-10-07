import numpy as np
import unittest

from sampy.pandas_xs.pandas_xs import DataFrameXS
from sampy.utils.decorators import use_debug_mode
from sampy.disease.single_species.jit_compiled_functions import (conditional_count_return_full_array,
                                                                 base_conditional_count_nb_agent_per_vertex,
                                                                 base_contaminate_vertices,
                                                                 transition_initialize_counters_of_newly_infected,
                                                                 transition_conditional_count_nb_agent_per_vertex,
                                                                 transition_falsify_when_condition)


use_debug_mode(DataFrameXS)


class TestJitCompiledFuncSingleSpeciesDisease(unittest.TestCase):
    def test_conditional_count_return_full_array(self):
        nb_vertex = 5
        arr_pos = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4])
        condition = np.array([False, False, True, True, True, False, True, False, True])
        count = conditional_count_return_full_array(nb_vertex, arr_pos, condition)
        self.assertTrue((count == np.array([0, 0, 2, 2, 2, 2, 2, 1, 1])).all())

    def test_base_conditional_count_nb_agent_per_vertex(self):
        nb_vertex = 5
        arr_pos = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4])
        condition = np.array([False, False, True, True, True, False, True, False, True])
        count = base_conditional_count_nb_agent_per_vertex(condition, arr_pos, nb_vertex)
        self.assertTrue((count == np.array([0, 0, 2, 2, 1])).all())

    def test_base_contaminate_vertices(self):
        new_infected = np.array([True, True, False, False])
        rand = np.array([0.1, 0.2, 0.05, 0.4])
        level = 0.15
        base_contaminate_vertices(new_infected, rand, level)
        self.assertTrue((new_infected == np.array([True, False, False, False])).all())

    def test_transition_initialize_counters_of_newly_infected(self):
        new_infected = np.array([True, False, True, False])
        counts = np.array([0, 17, 3, 14])
        new_counts = np.array([4, 8])
        transition_initialize_counters_of_newly_infected(new_infected, counts, new_counts)
        self.assertTrue((counts == np.array([4, 17, 8, 14])).all())

    def test_transition_conditional_count_nb_agent_per_vertex(self):
        nb_vertex = 5
        arr_pos = np.array([1, 1, 2, 2, 3, 3, 3, 4, 4])
        condition = np.array([False, False, True, True, True, False, True, False, True])
        count = transition_conditional_count_nb_agent_per_vertex(condition, arr_pos, nb_vertex)
        self.assertTrue((count == np.array([0, 0, 2, 2, 1])).all())

    def test_transition_falsify_when_condition(self):
        to_falsify = np.array([True, True, False, False, True])
        condition = np.array([True, False, True, False, False, False])
        transition_falsify_when_condition(to_falsify, condition)
        self.assertTrue((to_falsify == np.array([False, True, False, False, True])).all())
