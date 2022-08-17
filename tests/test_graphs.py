import unittest

from sampy.graph.topology import BaseTopology
from sampy.graph.topology import SquareGridWithDiagTopology
from sampy.graph.vertex_attributes import BaseVertexAttributes
from sampy.utils.decorators import sampy_class, use_debug_mode
from sampy.pandas_xs.pandas_xs import DataFrameXS

import numpy as np


use_debug_mode(DataFrameXS)
use_debug_mode(BaseTopology)
use_debug_mode(SquareGridWithDiagTopology)
use_debug_mode(BaseVertexAttributes)


class TestBaseTopology(unittest.TestCase):
    def test_creation_object(self):
        current_object = BaseTopology()
        self.assertTrue(hasattr(current_object, 'connections'))
        self.assertTrue(hasattr(current_object, 'weights'))
        self.assertTrue(hasattr(current_object, 'type'))
        self.assertTrue(hasattr(current_object, 'dict_cell_id_to_ind'))
        self.assertTrue(hasattr(current_object, 'time'))
        self.assertTrue(hasattr(current_object, 'on_ticker'))

        self.assertEqual(current_object.time, 0)
        self.assertEqual(current_object.dict_cell_id_to_ind, {})
        self.assertEqual(current_object.on_ticker, ['increment_time'])

    def test_increment_time(self):
        current_object = BaseTopology()
        current_object.increment_time()
        self.assertEqual(current_object.time, 1)
        current_object.increment_time()
        self.assertEqual(current_object.time, 2)

    def test_tick(self):
        current_object = BaseTopology()
        current_object.increment_time()
        self.assertEqual(current_object.time, 1)
        current_object.increment_time()
        self.assertEqual(current_object.time, 2)

        # we create an object with another method added to the 'on_ticker' list, and then test tick
        class TickTest(BaseTopology):
            def meth(self):
                self.meth = True

        current_object = TickTest()
        current_object.on_ticker.append('meth')
        current_object.tick()
        self.assertEqual(current_object.time, 1)
        self.assertTrue(current_object.meth)


class TestSquareGridWithDiagTopology(unittest.TestCase):
    def test_object_creation(self):
        # we need to turn the topology into a sampy class, in order to have the init of BaseTopology to be executed
        # first. That's why we create the following container
        @sampy_class
        class Graph(SquareGridWithDiagTopology):
            def __init__(self, **kwargs):
                pass

        with self.assertRaises(ValueError):
            Graph()

        current_object = Graph(shape=(3, 3))
        self.assertEqual(current_object.time, 0)
        self.assertEqual(current_object.shape, (3, 3))
        self.assertEqual(current_object.type, 'SquareGridWithDiag')
        self.assertEqual(current_object.number_vertices, 9)

        target_connections = np.array([[1, 3, 4, -1, -1, -1, -1, -1],
                                       [0, 2, 3, 4, 5, -1, -1, -1],
                                       [1, 4, 5, -1, -1, -1, -1, -1],
                                       [0, 1, 4, 6, 7, -1, -1, -1],
                                       [0, 1, 2, 3, 5, 6, 7, 8],
                                       [1, 2, 4, 7, 8, -1, -1, -1],
                                       [3, 4, 7, -1, -1, -1, -1, -1],
                                       [3, 4, 5, 6, 8, -1, -1, -1],
                                       [4, 5, 7, -1, -1, -1, -1, -1]])
        self.assertTrue((target_connections == current_object.connections).all())


class TestBaseVertexAttributes(unittest.TestCase):
    def test_object_creation(self):
        current_object = BaseVertexAttributes()
        self.assertTrue(hasattr(current_object, 'df_attributes'))
        self.assertIsInstance(current_object.df_attributes, DataFrameXS)

    def test_add_cst_attributes(self):
        current_object = BaseVertexAttributes()
        current_object.connections = np.full((10, 2), 0)
        current_object.weights = np.full((10, 2), 0)

        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute('test', {})
        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute('test', 'hihi')

        current_object.create_vertex_attribute('test', 10)
        self.assertTrue('test' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test'] == np.full((10,), 10)).all())

        current_object.create_vertex_attribute('test2', 1.)
        self.assertTrue('test2' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test2'] == np.full((10,), 1.)).all())

        current_object.create_vertex_attribute('test', 5)
        self.assertTrue('test' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test'] == np.full((10,), 5)).all())

    def test_add_array_attributes(self):
        current_object = BaseVertexAttributes()
        current_object.connections = np.full((10, 2), 0)
        current_object.weights = np.full((10, 2), 0)

        with self.assertRaises(TypeError):
            current_object.create_vertex_attribute(17, [{} for _ in range(10)])
        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute('test', [{} for _ in range(10)])
        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute('test', [str(i) for i in range(10)])
        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute('test', [0, 1, 2])
        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute('test', [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0], [0, 0],
                                                            [0, 0], [0, 0], [0, 0]])

        current_object.create_vertex_attribute('test', [i for i in range(10)])
        self.assertTrue('test' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test'] == np.array([i for i in range(10)])).all())

        current_object.create_vertex_attribute('test2', [float(i) for i in range(10)])
        self.assertTrue('test2' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test2'] == np.array([float(i) for i in range(10)])).all())

        current_object.create_vertex_attribute('test', [i + 1 for i in range(10)])
        self.assertTrue('test' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test'] == np.array([i + 1 for i in range(10)])).all())

    def test_create_vertex_attribute_from_dict(self):
        current_object = BaseVertexAttributes()
        current_object.connections = np.full((10, 2), 0)
        current_object.weights = np.full((10, 2), 0)
        current_object.dict_cell_id_to_ind = {str(i): i for i in range(10)}
        current_object.number_vertices = 10

        with self.assertRaises(TypeError):
            current_object.create_vertex_attribute_from_dict(17, {})
        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute_from_dict('test', [])
        with self.assertRaises(ValueError):
            current_object.create_vertex_attribute_from_dict('test', {1: 1})

        current_object.create_vertex_attribute_from_dict('test', {str(i): i+1 for i in range(8)}, -1)
        self.assertTrue('test' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test'] == np.array([1, 2, 3, 4, 5, 6, 7, 8, -1, -1])).all())

        current_object.create_vertex_attribute_from_dict('test2', {'3': 2., '7': 3.}, -1.)
        self.assertTrue('test2' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test2'] == np.array([-1., -1., -1., 2., -1., -1., -1.,
                                                                            3., -1., -1.])).all())

        current_object.create_vertex_attribute_from_dict('test', {str(i): i for i in range(10)}, 0)
        self.assertTrue('test' in current_object.df_attributes.dict_colname_to_index)
        self.assertTrue((current_object.df_attributes['test'] == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])).all())

    def test_change_type(self):
        current_object = BaseVertexAttributes()
        current_object.connections = np.full((10, 2), 0)
        current_object.weights = np.full((10, 2), 0)
        current_object.dict_cell_id_to_ind = {str(i): i for i in range(10)}
        current_object.number_vertices = 10

        current_object.create_vertex_attribute('test', 1)

        with self.assertRaises(TypeError):
            current_object.change_type_attribute(1, 'int8')
        with self.assertRaises(KeyError):
            current_object.change_type_attribute('wrong_name', 'int8')
        with self.assertRaises(ValueError):
            current_object.change_type_attribute('test', 'object')

        current_object.change_type_attribute('test', 'int8')
        self.assertEqual(str(current_object.df_attributes['test'].dtype), 'int8')

        current_object.create_vertex_attribute('test2', [i % 2 for i in range(10)])
        current_object.change_type_attribute('test2', 'bool')
        self.assertTrue((current_object.df_attributes['test2'] == np.array([i % 2 == 1 for i in range(10)])).all())
