import numpy as np
import unittest

from sampy.pandas_xs.pandas_xs import DataFrameXS
from sampy.utils.decorators import use_debug_mode
from sampy.disease.single_species.jit_compiled_functions import (conditional_count_return_full_array,
                                                                 base_conditional_count_nb_agent_per_vertex)


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
