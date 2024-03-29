import unittest

from sampy.graph.topology import BaseTopology
from sampy.graph.topology import SquareGridWithDiagTopology
from sampy.graph.vertex_attributes import BaseVertexAttributes
from sampy.utils.decorators import sampy_class, use_debug_mode
from sampy.pandas_xs.pandas_xs import DataFrameXS
from sampy.graph.builtin_graph import OrientedHexagonalLattice

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


class TestOrientedHexagonalLattice(unittest.TestCase):
    def test_object_creation(self):
        with self.assertRaises(ValueError):
            OrientedHexagonalLattice()

        with self.assertRaises(ValueError):
            OrientedHexagonalLattice(nb_hex_x_axis=2, nb_hex_y_axis=3)

        with self.assertRaises(ValueError):
            OrientedHexagonalLattice(nb_hex_x_axis=3, nb_hex_y_axis=2)

        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=3, nb_hex_y_axis=3)
        self.assertTrue(hasattr(hex_grid, 'df_attributes'))
        self.assertTrue(hasattr(hex_grid, 'connections'))
        self.assertTrue(hasattr(hex_grid, 'weights'))

        target_connections = np.array([[ 1,  3, -1, -1, -1, -1],
                                       [ 2,  4,  3,  0, -1, -1],
                                       [-1,  5,  4,  1, -1, -1],
                                       [ 4,  7,  6, -1,  0,  1],
                                       [ 5,  8,  7,  3,  1,  2],
                                       [-1, -1,  8,  4,  2, -1],
                                       [ 7, -1, -1, -1, -1,  3],
                                       [ 8, -1, -1,  6,  3,  4],
                                       [-1, -1, -1,  7,  4,  5]])
        
        self.assertTrue((hex_grid.connections == target_connections).all())

        self.assertTrue(((hex_grid.connections >= 0) == (hex_grid.weights > 0.)).all())
    
        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=4, nb_hex_y_axis=3)
        self.assertTrue(hasattr(hex_grid, 'df_attributes'))
        self.assertTrue(hasattr(hex_grid, 'connections'))
        self.assertTrue(hasattr(hex_grid, 'weights'))

        target_connections = np.array([[ 1,  3, -1, -1, -1, -1],
                                       [ 2,  4,  3,  0, -1, -1],
                                       [-1,  5,  4,  1, -1, -1],
                                       [ 4,  7,  6, -1,  0,  1],
                                       [ 5,  8,  7,  3,  1,  2],
                                       [-1, -1,  8,  4,  2, -1],
                                       [ 7,  9, -1, -1, -1,  3],
                                       [ 8, 10,  9,  6,  3,  4],
                                       [-1, 11, 10,  7,  4,  5],
                                       [10, -1, -1, -1,  6,  7],
                                       [11, -1, -1,  9,  7,  8],
                                       [-1, -1, -1, 10,  8, -1]])
        
        self.assertTrue((hex_grid.connections == target_connections).all())

        self.assertTrue(((hex_grid.connections >= 0) == (hex_grid.weights > 0.)).all())

    def test_bigger_scale_even(self):
        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=6, nb_hex_y_axis=8)
        self.assertTrue(hex_grid.connections.shape == (48, 6))
        self.assertTrue(hex_grid.weights.shape == (48, 6))

        self.assertTrue(((hex_grid.connections >= 0) == (hex_grid.weights > 0.)).all())

        self.assertTrue((hex_grid.connections[38] == np.array([39, 46, 45, 37, 29, 30])).all())
        self.assertTrue((hex_grid.connections[19] == np.array([20, 27, 26, 18, 10, 11])).all())

        self.assertTrue((hex_grid.connections[47] == np.array([-1, -1, -1, 46, 39, -1])).all())
        self.assertTrue((hex_grid.connections[40] == np.array([41, -1, -1, -1, 32, 33])).all())
        self.assertTrue((hex_grid.connections[7] == np.array([-1, 15, 14, 6, -1, -1])).all())
        self.assertTrue((hex_grid.connections[0] == np.array([1, 8, -1, -1, -1, -1])).all())

    def test_bigger_scale_odd(self):
        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=7, nb_hex_y_axis=8)

        self.assertTrue(hex_grid.connections.shape == (56, 6))
        self.assertTrue(hex_grid.weights.shape == (56, 6))

        self.assertTrue(((hex_grid.connections >= 0) == (hex_grid.weights > 0.)).all())

        self.assertTrue((hex_grid.connections[38] == np.array([39, 46, 45, 37, 29, 30])).all())
        self.assertTrue((hex_grid.connections[19] == np.array([20, 27, 26, 18, 10, 11])).all())
        self.assertTrue((hex_grid.connections[45] == np.array([46, 54, 53, 44, 37, 38])).all())
        self.assertTrue((hex_grid.connections[47] == np.array([-1, -1, 55, 46, 39, -1])).all())

        self.assertTrue((hex_grid.connections[55] == np.array([-1, -1, -1, 54, 46, 47])).all())
        self.assertTrue((hex_grid.connections[48] == np.array([49, -1, -1, -1, -1, 40])).all())
        self.assertTrue((hex_grid.connections[7] == np.array([-1, 15, 14, 6, -1, -1])).all())
        self.assertTrue((hex_grid.connections[0] == np.array([1, 8, -1, -1, -1, -1])).all())

    def test_basic_spatial_components(self):
        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=3, nb_hex_y_axis=3)
        hex_grid.set_coords_from_vector((0, 0), (0, 1))
        self.assertTrue('coord_x' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue('coord_y' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue(hex_grid.df_attributes['coord_x'][0] == 0.)
        self.assertTrue(hex_grid.df_attributes['coord_y'][0] == 0.)

        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=100, nb_hex_y_axis=100)
        hex_grid.set_coords_from_vector((0, 0), (0, 1))
        self.assertTrue('coord_x' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue('coord_y' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue(hex_grid.df_attributes['coord_x'][0] == 0.)
        self.assertTrue(hex_grid.df_attributes['coord_y'][0] == 0.)

        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=3, nb_hex_y_axis=3)
        hex_grid.set_coords_from_vector((2.4, 3.5), (0, 1))
        self.assertTrue('coord_x' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue('coord_y' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue(hex_grid.df_attributes['coord_x'][0] == 2.4)
        self.assertTrue(hex_grid.df_attributes['coord_y'][0] == 3.5)

        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=3, nb_hex_y_axis=3)
        hex_grid.set_coords_from_vector((0, 2), (0, 1), 
                                        attribute_coord_x='beep',
                                        attribute_coord_y='boop')
        self.assertTrue('beep' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue('boop' in hex_grid.df_attributes.dict_colname_to_index)
        self.assertTrue(hex_grid.df_attributes['beep'][0] == 0.)
        self.assertTrue(hex_grid.df_attributes['boop'][0] == 2.)

    def test_advanced_spatial_components(self):
        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=3, nb_hex_y_axis=3)
        hex_grid.set_coords_from_vector((0, 0), (0, 1))

        target_x = np.array([0., 0., 0., 
                             0.8660254, 0.8660254, 0.8660254, 
                             1.73205081, 1.73205081, 1.73205081], 
                             dtype=np.float64)
        target_y = np.array([0.0, 1.0, 2.0, 
                             0.5, 1.5, 2.5, 
                             0., 1.0, 2.0],
                             dtype=np.float64)
        
        err_x = hex_grid.df_attributes['coord_x'] - target_x
        success = np.abs(err_x) < 1e-8
        self.assertTrue(success.all())

        err_y = hex_grid.df_attributes['coord_y'] - target_y
        success = np.abs(err_y) < 1e-8
        self.assertTrue(success.all())

        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=6, nb_hex_y_axis=8)
        hex_grid.set_coords_from_vector((0, 0), (0, 1))

        self.assertTrue(np.abs(hex_grid.df_attributes['coord_x'][0] - 0.) < 1e-8)
        self.assertTrue(np.abs(hex_grid.df_attributes['coord_y'][6] - 6.) < 1e-8)

        self.assertTrue(np.abs(hex_grid.df_attributes['coord_x'][25] - 3 * 0.8660254) < 1e-6)
        self.assertTrue(np.abs(hex_grid.df_attributes['coord_y'][25] - 1.5) < 1e-8)

    def test_hex_polygon_vertices(self):
        hex_grid = OrientedHexagonalLattice(nb_hex_x_axis=3, nb_hex_y_axis=3)
        hex_grid.set_coords_from_vector((0, 0), (0, 1))
        hex_grid.create_hexagon_vertices()

        target_vertices = np.array([[[-2.88675135e-01,  5.00000000e-01],
        [-5.77350269e-01,  1.11022302e-16],
        [-2.88675135e-01, -5.00000000e-01],
        [ 2.88675135e-01, -5.00000000e-01],
        [ 5.77350269e-01, -1.11022302e-16],
        [ 2.88675135e-01,  5.00000000e-01]],

       [[-2.88675135e-01,  1.50000000e+00],
        [-5.77350269e-01,  1.00000000e+00],
        [-2.88675135e-01,  5.00000000e-01],
        [ 2.88675135e-01,  5.00000000e-01],
        [ 5.77350269e-01,  1.00000000e+00],
        [ 2.88675135e-01,  1.50000000e+00]],

       [[-2.88675135e-01,  2.50000000e+00],
        [-5.77350269e-01,  2.00000000e+00],
        [-2.88675135e-01,  1.50000000e+00],
        [ 2.88675135e-01,  1.50000000e+00],
        [ 5.77350269e-01,  2.00000000e+00],
        [ 2.88675135e-01,  2.50000000e+00]],

       [[ 5.77350269e-01,  1.00000000e+00],
        [ 2.88675135e-01,  5.00000000e-01],
        [ 5.77350269e-01,  2.22044605e-16],
        [ 1.15470054e+00,  0.00000000e+00],
        [ 1.44337567e+00,  5.00000000e-01],
        [ 1.15470054e+00,  1.00000000e+00]],

       [[ 5.77350269e-01,  2.00000000e+00],
        [ 2.88675135e-01,  1.50000000e+00],
        [ 5.77350269e-01,  1.00000000e+00],
        [ 1.15470054e+00,  1.00000000e+00],
        [ 1.44337567e+00,  1.50000000e+00],
        [ 1.15470054e+00,  2.00000000e+00]],

       [[ 5.77350269e-01,  3.00000000e+00],
        [ 2.88675135e-01,  2.50000000e+00],
        [ 5.77350269e-01,  2.00000000e+00],
        [ 1.15470054e+00,  2.00000000e+00],
        [ 1.44337567e+00,  2.50000000e+00],
        [ 1.15470054e+00,  3.00000000e+00]],

       [[ 1.44337567e+00,  5.00000000e-01],
        [ 1.15470054e+00,  4.44089210e-16],
        [ 1.44337567e+00, -5.00000000e-01],
        [ 2.02072594e+00, -5.00000000e-01],
        [ 2.30940108e+00,  2.22044605e-16],
        [ 2.02072594e+00,  5.00000000e-01]],

       [[ 1.44337567e+00,  1.50000000e+00],
        [ 1.15470054e+00,  1.00000000e+00],
        [ 1.44337567e+00,  5.00000000e-01],
        [ 2.02072594e+00,  5.00000000e-01],
        [ 2.30940108e+00,  1.00000000e+00],
        [ 2.02072594e+00,  1.50000000e+00]],

       [[ 1.44337567e+00,  2.50000000e+00],
        [ 1.15470054e+00,  2.00000000e+00],
        [ 1.44337567e+00,  1.50000000e+00],
        [ 2.02072594e+00,  1.50000000e+00],
        [ 2.30940108e+00,  2.00000000e+00],
        [ 2.02072594e+00,  2.50000000e+00]]])

        diff_vert = hex_grid.cell_vertices - target_vertices
        self.assertTrue((np.abs(diff_vert) < 1e-8).all())