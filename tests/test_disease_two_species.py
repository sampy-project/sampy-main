import numpy as np
import unittest

from sampy.pandas_xs.pandas_xs import DataFrameXS
from sampy.utils.decorators import use_debug_mode
from sampy.disease.two_species.jit_compiled_functions import base_contaminate_vertices

use_debug_mode(DataFrameXS)


class TestJitCompiledFuncTwoSpeciesDisease(unittest.TestCase):
    def test_base_contaminate_vertices(self):
        new_infected = np.array([True, True, False, False])
        rand = np.array([0.1, 0.2, 0.05, 0.4])
        level = 0.15
        base_contaminate_vertices(new_infected, rand, level)
        self.assertTrue((new_infected == np.array([True, False, False, False])).all())
