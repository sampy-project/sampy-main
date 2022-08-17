from sampy.agent.base import BaseAgingAgent
from sampy.agent.builtin_agent import BasicMammal
from sampy.agent.jit_compiled_functions import (count_nb_agent_per_vertex,
                                                conditional_count_nb_agent_per_vertex,
                                                mortality_natural_death_orm_methodology,
                                                mortality_natural_death_orm_methodology_condition_death,
                                                mortality_natural_death_orm_methodology_condition_count,
                                                mortality_natural_death_orm_methodology_both_cond,
                                                reproduction_find_random_mate_on_position,
                                                reproduction_find_random_mate_on_position_condition,
                                                reproduction_find_random_mate_on_position_polygamous,
                                                reproduction_find_random_mate_on_position_polygamous_condition,
                                                movement_change_territory_and_position,
                                                movement_change_territory_and_position_condition,
                                                movement_mov_around_territory_fill_bool_mov_using_condition,
                                                movement_mov_around_territory,
                                                movement_dispersion_with_varying_nb_of_steps,
                                                movement_dispersion_with_varying_nb_of_steps_condition)
from sampy.pandas_xs.pandas_xs import DataFrameXS
from sampy.utils.decorators import use_debug_mode
import numpy as np
import unittest

use_debug_mode(DataFrameXS)
use_debug_mode(BaseAgingAgent)


class TestJitCompiledFuncAgent(unittest.TestCase):
    def test_count_nb_agents_per_vertex(self):
        pos = np.array([0, 0, 0, 1, 1, 3, 4, 3, 2, 0, 2])
        nb_vertex = 6
        x = count_nb_agent_per_vertex(pos, nb_vertex)
        self.assertTrue((x == np.array([4, 2, 2, 2, 1, 0])).all())

        pos = np.array([0 for _ in range(1000)])
        nb_vertex = 3
        x = count_nb_agent_per_vertex(pos, nb_vertex)
        self.assertTrue((x == np.array([1000, 0, 0])).all())

    def test_conditional_count_nb_agents_per_vertex(self):
        pos = np.array([0, 0, 0, 1, 1, 3, 4, 3, 2, 0, 2])
        condition = np.array([False, False, False, True, True, True, True, True, True, False, True])
        nb_vertex = 6
        x = conditional_count_nb_agent_per_vertex(condition, pos, nb_vertex)
        self.assertTrue((x == np.array([0, 2, 2, 2, 1, 0])).all())

        pos = np.array([0 for _ in range(1000)])
        condition = np.array([True]*400 + [False]*600)
        nb_vertex = 3
        x = conditional_count_nb_agent_per_vertex(condition, pos, nb_vertex)
        self.assertTrue((x == np.array([400, 0, 0])).all())

    def test_mortality_natural_death_orm_methodology(self):
        bias = 0.
        rand = np.array([0.5 for _ in range(8)])
        prob_male = np.array([0.2, 0.5, 0.8, 1.])
        prob_female = np.array([0.2, 0.4, 0.8, 1.])
        count = np.array([2, 2, 4])
        pos = np.array([0, 1, 2, 2, 0, 1, 2, 2])
        k = np.array([2, 2, 2])
        age = np.array([1, 0, 1, 1, 2, 1, 3, 3])
        gender = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        expected_death = [0.5 > float(prob_male[1]) * (float(count[0]) / float(k[0])),
                          0.5 > float(prob_male[0]) * (float(count[1]) / float(k[1])),
                          0.5 > float(prob_male[1]) * (float(count[2]) / float(k[2])),
                          0.5 > float(prob_female[1]) * (float(count[2] - 1) / float(k[2])),
                          0.5 > float(prob_female[2]) * (float(count[0] - 1) / float(k[2])),
                          0.5 > float(prob_female[1]) * (float(count[1]) / float(k[1])),
                          0.5 > float(prob_female[3]) * (float(count[2] - 2) / float(k[2])),
                          0.5 > float(prob_male[3]) * (float(count[2] - 3) / float(k[2]))]
        death = mortality_natural_death_orm_methodology(bias, rand, prob_male, prob_female, count, pos, k, age, gender)
        expected_death = np.array(expected_death)
        self.assertTrue((death == expected_death).all())

    def test_mortality_natural_death_orm_methodology_condition_death(self):
        bias = 0.
        rand = np.array([0.5 for _ in range(8)])
        prob_male = np.array([0.2, 0.5, 0.8, 1.])
        prob_female = np.array([0.2, 0.4, 0.8, 1.])
        count = np.array([2, 2, 4])
        pos = np.array([0, 1, 2, 2, 0, 1, 2, 2])
        k = np.array([2, 2, 2])
        age = np.array([1, 0, 1, 1, 2, 1, 3, 3])
        gender = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        cond_death = np.array([True, True, True, True, True, True, False, True])
        expected_death = [0.5 > float(prob_male[1]) * (float(count[0]) / float(k[0])),
                          0.5 > float(prob_male[0]) * (float(count[1]) / float(k[1])),
                          0.5 > float(prob_male[1]) * (float(count[2]) / float(k[2])),
                          0.5 > float(prob_female[1]) * (float(count[2] - 1) / float(k[2])),
                          0.5 > float(prob_female[2]) * (float(count[0] - 1) / float(k[2])),
                          0.5 > float(prob_female[1]) * (float(count[1]) / float(k[1])),
                          True,
                          0.5 > float(prob_male[3]) * (float(count[2] - 2) / float(k[2]))]
        expected_death = np.array(expected_death, dtype=np.bool_)
        death = mortality_natural_death_orm_methodology_condition_death(bias, rand, prob_male, prob_female, count,
                                                                        cond_death, pos, k, age, gender)
        self.assertTrue((death == expected_death).all())

    def test_mortality_natural_death_orm_methodology_condition_count(self):
        # note that with the values considered below, the 4th agent should die if the 7th agent was counted.
        bias = 0.
        rand = np.array([0.5 for _ in range(8)])
        prob_male = np.array([0.2, 0.5, 0.8, 1.])
        prob_female = np.array([0.2, 0.4, 0.8, 1.])
        count = np.array([2, 2, 3])
        pos = np.array([0, 1, 2, 2, 0, 1, 2, 2])
        k = np.array([2, 2, 2])
        age = np.array([1, 0, 1, 1, 2, 1, 3, 3])
        gender = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        cond_count = np.array([True, True, True, True, True, True, False, True])
        expected_death = [0.5 > float(prob_male[1]) * (float(count[0]) / float(k[0])),
                          0.5 > float(prob_male[0]) * (float(count[1]) / float(k[1])),
                          0.5 > float(prob_male[1]) * (float(count[2]) / float(k[2])),
                          0.5 > float(prob_female[1]) * (float(count[2] - 1) / float(k[2])),
                          0.5 > float(prob_female[2]) * (float(count[0] - 1) / float(k[0])),
                          0.5 > float(prob_female[1]) * (float(count[1]) / float(k[1])),
                          0.5 > float(prob_female[3]) * (float(count[2] - 2) / float(k[2])),
                          0.5 > float(prob_male[3]) * (float(count[2] - 2) / float(k[2]))]
        expected_death = np.array(expected_death, dtype=np.bool_)
        death = mortality_natural_death_orm_methodology_condition_count(bias, rand, prob_male, prob_female, count,
                                                                        cond_count, pos, k, age, gender)
        self.assertTrue((death == expected_death).all())

    def test_mortality_natural_death_orm_methodology_both_cond(self):
        # note that with the values considered below, the 4th agent should die if the 7th agent was counted.
        bias = 0.
        rand = np.array([0.5 for _ in range(8)])
        prob_male = np.array([0.2, 0.5, 0.8, 1.])
        prob_female = np.array([0.2, 0.4, 0.8, 1.])
        count = np.array([2, 2, 3])
        pos = np.array([0, 1, 2, 2, 0, 1, 2, 2])
        k = np.array([2, 2, 2])
        age = np.array([1, 0, 1, 1, 2, 1, 3, 3])
        gender = np.array([0, 0, 0, 1, 1, 1, 1, 0])
        cond_count = np.array([True, True, True, True, True, True, False, True])
        cond_death = np.array([True, True, True, True, True, True, True, False])
        expected_death = [0.5 > float(prob_male[1]) * (float(count[0]) / float(k[0])),
                          0.5 > float(prob_male[0]) * (float(count[1]) / float(k[1])),
                          0.5 > float(prob_male[1]) * (float(count[2]) / float(k[2])),
                          0.5 > float(prob_female[1]) * (float(count[2] - 1) / float(k[2])),
                          0.5 > float(prob_female[2]) * (float(count[0] - 1) / float(k[0])),
                          0.5 > float(prob_female[1]) * (float(count[1]) / float(k[1])),
                          0.5 > float(prob_female[3]) * (float(count[2] - 2) / float(k[2])),
                          True]
        expected_death = np.array(expected_death, dtype=np.bool_)
        death = mortality_natural_death_orm_methodology_both_cond(bias, rand, prob_male, prob_female, count, cond_death,
                                                                  cond_count, pos, k, age, gender)
        self.assertTrue((death == expected_death).all())

    def test_reproduction_find_random_mate_on_position(self):
        col_mate = np.array([i for i in range(10)])
        col_pregnancy = np.array([True for _ in range(10)])
        arr_id = np.array([i for i in range(10)])
        position = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        gender = np.array([i % 2 for i in range(10)])
        nb_vertex = 5
        rand_preg = np.array([0.4, 0.3, 0.2, 0.1, 0.8])
        prob_preg = 0.5
        reproduction_find_random_mate_on_position(col_mate, col_pregnancy, arr_id, position, gender, nb_vertex,
                                                  rand_preg, prob_preg)
        self.assertTrue((col_mate == np.array([1, 0, 3, 2, -1, -1, 7, 6, 9, 8])).all())
        self.assertTrue((col_pregnancy == np.array([False, True, False, True, False,
                                                    False, False, True, False, False])).all())

    def test_reproduction_find_random_mate_on_position_condition(self):
        col_mate = np.array([-2 for _ in range(10)])
        col_pregnancy = np.array([True for _ in range(10)])
        arr_id = np.array([i for i in range(10)])
        position = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        gender = np.array([i % 2 for i in range(10)])
        nb_vertex = 5
        rand_preg = np.array([0.4, 0.3, 0.2, 0.1, 0.8])
        prob_preg = 0.5
        condition = np.array([False] + [True for _ in range(9)])
        reproduction_find_random_mate_on_position_condition(col_mate,col_pregnancy, arr_id, position, gender, nb_vertex,
                                                            rand_preg, prob_preg, condition)
        self.assertTrue((col_mate == np.array([-2, -1, 3, 2, -1, -1, 7, 6, 9, 8])).all())
        self.assertTrue((col_pregnancy == np.array([True, False, False, True, False,
                                                    False, False, True, False, False])).all())
        # Yup, there is a pregnant male above. That's normal. It's to check that the function is not modifying anything
        # when condition is set on False.

    def test_reproduction_find_random_mate_on_position_polygamous(self):
        col_mate = np.array([-2 for _ in range(10)])
        col_pregnancy = np.array([False for _ in range(10)])
        arr_id = np.array([i for i in range(10)])
        position = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        gender = np.array([i % 2 for i in range(10)])
        nb_vertex = 5
        rand_preg = np.array([0.4, 0.3, 0.2, 0.1, 0.8])
        rand_mate = np.array([0.98, 0.6, 0.1, 0.2, 0.49])
        prob_preg = 0.5
        reproduction_find_random_mate_on_position_polygamous(arr_id, position, gender, col_mate, col_pregnancy,
                                                             nb_vertex, rand_preg, rand_mate, prob_preg)
        self.assertTrue((col_mate == np.array([-1, 0, -1, 4, -1, -1, -1, 6, -1, 6])).all())
        self.assertTrue((col_pregnancy == np.array([False, True, False, True, False,
                                                    False, False, True, False, False])).all())

    def test_reproduction_find_random_mate_on_position_polygamous_condition(self):
        col_mate = np.array([-2 for _ in range(10)])
        col_pregnancy = np.array([True for _ in range(10)])
        arr_id = np.array([i for i in range(10)])
        position = np.array([0, 0, 1, 1, 1, 2, 3, 3, 3, 3])
        gender = np.array([i % 2 for i in range(10)])
        nb_vertex = 5
        rand_preg = np.array([0.4, 0.3, 0.2, 0.1, 0.8])
        rand_mate = np.array([0.98, 0.6, 0.1, 0.2, 0.49])
        prob_preg = 0.5
        condition = np.array([False] + [True for _ in range(9)])
        reproduction_find_random_mate_on_position_polygamous_condition(arr_id, position, gender, col_mate,
                                                                       col_pregnancy, nb_vertex, rand_preg, rand_mate,
                                                                       prob_preg, condition)
        self.assertTrue((col_mate == np.array([-2, -1, -1, 4, -1, -1, -1, 6, -1, 6])).all())
        self.assertTrue((col_pregnancy == np.array([True, False, False, True, False,
                                                    False, False, True, False, False])).all())

    def test_movement_change_territory_and_position(self):
        territory = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        position = np.array([i % 5 for i in range(12)])
        rand = np.array([0.5 for _ in range(12)])
        connections = np.array([[-1, -1, -1],
                                [2, -1, -1],
                                [1, -1, -1],
                                [4, 5, -1],
                                [3, 5, -1],
                                [3, -1, 4]])
        weights = np.array([[-1., -1., -1.],
                            [1., -1., -1.],
                            [1., -1., -1.],
                            [.6, 1., -1],
                            [.4, 1., -1.],
                            [.4, -1., 1.]])

        movement_change_territory_and_position(territory, position, rand, connections, weights)

        self.assertTrue((territory == position).all())
        self.assertTrue((territory == np.array([0, 0, 2, 2, 1, 1, 4, 4, 5, 5, 4, 4])).all())

    def test_movement_change_territory_and_position_condition(self):
        condition = np.array([i % 2 == 1 for i in range(12)])
        territory = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        position = np.array([i % 5 for i in range(12)])
        rand = np.array([0.5 for _ in range(12)])
        connections = np.array([[-1, -1, -1],
                                [2, -1, -1],
                                [1, -1, -1],
                                [4, 5, -1],
                                [3, 5, -1],
                                [3, -1, 4]])
        weights = np.array([[-1., -1., -1.],
                            [1., -1., -1.],
                            [1., -1., -1.],
                            [.6, 1., -1],
                            [.4, 1., -1.],
                            [.4, -1., 1.]])

        movement_change_territory_and_position_condition(territory, position, condition, rand, connections, weights)

        self.assertTrue((position == np.array([0, 0, 2, 2, 4, 1, 1, 4, 3, 5, 0, 4])).all())
        self.assertTrue((territory == np.array([0, 0, 1, 2, 2, 1, 3, 4, 4, 5, 5, 4])).all())

    def test_movement_mov_around_territory_fill_bool_mov_using_condition(self):
        condition = np.array([i % 2 == 0 for i in range(10)])
        pre_bol_mov = np.array([True, True, False, False, True])
        result = movement_mov_around_territory_fill_bool_mov_using_condition(pre_bol_mov, condition)

        self.assertTrue((result == np.array([True, False, True, False, False, False, False, False, True, False])).all())

    def test_movement_mov_around_territory(self):
        territory = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        position = np.array([i % 5 for i in range(12)])
        rand = np.array([0.5 for _ in range(6)])
        connections = np.array([[-1, -1, -1],
                                [2, -1, -1],
                                [1, -1, -1],
                                [4, 5, -1],
                                [3, 5, -1],
                                [3, -1, 4]])
        weights = np.array([[-1., -1., -1.],
                            [1., -1., -1.],
                            [1., -1., -1.],
                            [.6, 1., -1.],
                            [.4, 1., -1.],
                            [.4, -1., 1.]])
        bool_mov = np.array([i % 2 == 0 for i in range(12)])
        movement_mov_around_territory(territory, position, bool_mov, rand, connections, weights)
        self.assertTrue((position == np.array([0, 1, 2, 3, 1, 0, 4, 2, 5, 4, 4, 1])).all())
        self.assertTrue((territory == np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])).all())

    def test_movement_dispersion_with_varying_nb_of_steps(self):
        territory = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        position = np.array([i % 5 for i in range(12)])
        nb_steps = np.array([0, 1, 1, 2, 1, 2, 1, 1, 0, 1, 0, 3])
        rand = np.array([.5, .5, .5, .5, .5, .5, .5, .7, .5, .5, .3, .5, .5])
        connections = np.array([[-1, -1, -1],
                                [2, -1, -1],
                                [1, -1, -1],
                                [4, 5, -1],
                                [3, 5, -1],
                                [3, -1, 4]])
        weights = np.array([[-1., -1., -1.],
                            [1., -1., -1.],
                            [1., -1., -1.],
                            [.6, 1., -1.],
                            [.4, 1., -1.],
                            [.4, -1., 1.]])
        movement_dispersion_with_varying_nb_of_steps(territory, position, rand, nb_steps, connections, weights)
        self.assertTrue((position == territory).all())
        self.assertTrue((position == np.array([0, 0, 2, 1, 1, 2, 5, 4, 4, 5, 5, 5])).all())

    def test_movement_dispersion_with_varying_nb_of_steps_condition(self):
        territory = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])
        position = np.array([i % 5 for i in range(12)])
        condition = np.array([True, False] + [True for _ in range(10)])
        nb_steps = np.array([0, 1, 2, 1, 2, 1, 1, 0, 1, 0, 3])
        rand = np.array([.5, .5, .5, .5, .5, .5, .7, .5, .5, .3, .5, .5])
        connections = np.array([[-1, -1, -1],
                                [2, -1, -1],
                                [1, -1, -1],
                                [4, 5, -1],
                                [3, 5, -1],
                                [3, -1, 4]])
        weights = np.array([[-1., -1., -1.],
                            [1., -1., -1.],
                            [1., -1., -1.],
                            [.6, 1., -1.],
                            [.4, 1., -1.],
                            [.4, -1., 1.]])
        movement_dispersion_with_varying_nb_of_steps_condition(territory, position, condition, rand, nb_steps,
                                                               connections, weights)
        self.assertTrue((position == np.array([0, 1, 2, 1, 1, 2, 5, 4, 4, 5, 5, 5])).all())
        self.assertTrue((territory == np.array([0, 0, 2, 1, 1, 2, 5, 4, 4, 5, 5, 5])).all())


class TestBaseAgingAgent(unittest.TestCase):

    def test_error_when_missing_graph(self):
        with self.assertRaises(ValueError):
            x = BaseAgingAgent()

    def test_creation_when_something_as_graph(self):
        x = BaseAgingAgent(graph='test')
        self.assertTrue(isinstance(x.df_population, DataFrameXS))
        self.assertEqual(x.df_population.list_col_name, ['col_id', 'age'])
        self.assertEqual(x.df_population.list_col, [None, None])
        self.assertEqual(x.df_population.nb_rows, 0)
        self.assertEqual(x.df_population.nb_cols, 2)
        self.assertEqual(x.df_population.shape, (0, 2))
        self.assertEqual(x.dict_default_val, {'age': 0})

    def test_change_default_val(self):
        x = BaseAgingAgent(graph='test')
        with self.assertRaises(TypeError):
            x.set_default_val([])
        with self.assertRaises(KeyError):
            x.set_default_val({'test': 10})

        x.add_attribute('test', def_value=10)
        self.assertEqual(x.dict_default_val, {'age': 0, 'test': 10})
        x.set_default_val({'age': 2}, replace=True)
        self.assertEqual(x.dict_default_val, {'age': 2})

    def test_add_agents(self):
        x = BaseAgingAgent(graph='test')
        with self.assertRaises(ValueError):
            x.add_agents({'age': ['test', 'test2']})
        with self.assertRaises(KeyError):
            x.add_agents({'wrong_col_name': [0, 0]})

        x.add_agents({'age': [1, 2, 3, 4]})
        self.assertTrue((x.df_population['age'] == np.array([1, 2, 3, 4])).all())
        self.assertTrue((x.df_population['col_id'] == np.array([0, 1, 2, 3])).all())
        self.assertEqual(x.df_population.shape, (4, 2))

        x.add_attribute('test_attr', def_value=10)
        x.add_agents({'age': 1})
        self.assertTrue((x.df_population['age'] == np.array([1, 2, 3, 4, 1])).all())
        self.assertTrue((x.df_population['col_id'] == np.array([0, 1, 2, 3, 4])).all())
        self.assertTrue((x.df_population['test_attr'] == np.array([10, 10, 10, 10, 10])).all())
        self.assertEqual(x.df_population.shape, (5, 3))

        x.add_attribute('test_attr2')
        x.add_agents({'age': 17})
        self.assertTrue((x.df_population['age'] == np.array([1, 2, 3, 4, 1, 17])).all())
        self.assertTrue((x.df_population['col_id'] == np.array([0, 1, 2, 3, 4, 5])).all())
        self.assertTrue((x.df_population['test_attr'] == np.array([10, 10, 10, 10, 10, 10])).all())
        self.assertTrue((np.nan_to_num(x.df_population['test_attr2'], 0.) == np.array([0., 0., 0., 0., 0., 0.])).all())
        self.assertEqual(x.df_population.shape, (6, 4))

        x.add_agents({'age': [3, 4], 'test_attr2': 3.})
        self.assertTrue((x.df_population['age'] == np.array([1, 2, 3, 4, 1, 17, 3, 4])).all())
        self.assertTrue((x.df_population['col_id'] == np.array([0, 1, 2, 3, 4, 5, 6, 7])).all())
        self.assertTrue((x.df_population['test_attr'] == np.array([10, 10, 10, 10, 10, 10, 10, 10])).all())
        self.assertTrue((np.nan_to_num(x.df_population['test_attr2'], 0.) == np.array([0., 0., 0., 0., 0., 0., 3., 3.])).all())
        self.assertEqual(x.df_population.shape, (8, 4))

        with self.assertRaises(ValueError):
            x.add_agents({'age': [1, 2], 'test_attr': [4, 5, 6]})

    def test_count_agents_per_vertex(self):
        class FakeGraph:
            weights = np.array([0, 0, 0, 0, 0])
        x = BaseAgingAgent(graph=FakeGraph())
        x.add_attribute('position', def_value=0)
        x.add_agents({'age': 0, 'position': [0, 0, 0, 1, 1, 3, 4, 3, 2, 0, 2]})

        count_all = x.count_pop_per_vertex()

        self.assertEqual(count_all.sum(), 11)
        self.assertTrue((count_all == np.array([4, 2, 2, 2, 1])).all())

        with self.assertRaises(ValueError):
            condition = np.array([True, False])
            x.count_pop_per_vertex(condition=condition)

        with self.assertRaises(ValueError):
            condition = np.array([0, 0, 0, 1, 1, 3, 4, 3, 2, 0, 2])
            x.count_pop_per_vertex(condition=condition)

        with self.assertRaises(ValueError):
            condition = np.array([[0, 0, 0, 1, 1, 3, 4, 3, 2, 0, 2]])
            x.count_pop_per_vertex(condition=condition)

        with self.assertRaises(ValueError):
            condition = {}
            x.count_pop_per_vertex(condition=condition)

        condition = np.array([False, False, False, True, True, True, True, True, True, False, True])
        count_cond = x.count_pop_per_vertex(condition=condition)

        self.assertEqual(count_cond.sum(), 7)
        self.assertTrue((count_cond == np.array([0, 2, 2, 2, 1])).all())


class TestMovementTerritorialMovementWithoutResistance(unittest.TestCase):
    pass


class TestBuiltInAgentBasicMammal(unittest.TestCase):
    def test_object_creation(self):
        with self.assertRaises(ValueError):
            x = BasicMammal()
        list_expected_col = ['col_id', 'age', 'position', 'territory', 'gender', 'mom_id', 'dad_id', 'is_pregnant']