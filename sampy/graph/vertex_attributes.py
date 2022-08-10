import numpy as np
from .jit_compiled_functions import compute_sin_attr_with_condition
from ..pandas_xs.pandas_xs import DataFrameXS


class BaseVertexAttributes:
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_attributes'):
            self.df_attributes = DataFrameXS()

    def _sampy_debug_create_vertex_attribute(self, attr_name, value):
        if not isinstance(attr_name, str):
            raise TypeError("the name of a vertex attribute should be a string.")
        arr = np.array(value)
        if len(arr.shape) != 0:
            if len(arr.shape) > 1:
                raise ValueError('Shape of provided array for graph attribute ' + attr_name + ' is ' + str(arr.shape) +
                                 ', while Sampy expects an array of shape (' + str(self.weights.shape[0]) +
                                 ',).')
            if arr.shape[0] != self.weights.shape[0]:
                raise ValueError('Provided array for graph attribute ' + attr_name + ' has ' +
                                 str(arr.shape[0]) + 'elements, while the graph has ' + str(self.weights.shape[0]) +
                                 'vertices. Those numbers should be the same.')

    def create_vertex_attribute(self, attr_name, value):
        """
        Creates a new vertex attribute and populates its values. Accepted input for 'value' are:
                - None: in this case, the attribute column is set empty
                - A single value, in which case all vertexes will have the same attribute value
                - A 1D array, which will become the attribute column.

        Note that if you use a 1D array, then you are implicitly working with the indexes of the vertices, that is that
        the value at position 'i' in the array corresponds to the attribute value associated with the vertex whose index
        is 'i'. If you want to work with vertexes id instead, use the method 'create_vertex_attribute_from_dict'.

        :param attr_name: string, name of the attribute
        :param value: either None, a single value, or a 1D array.
        """
        if self.df_attributes.nb_rows == 0 and len(np.array(value).shape) == 0:
            self.df_attributes[attr_name] = [value for _ in range(self.weights.shape[0])]
        else:
            self.df_attributes[attr_name] = value

    def _sampy_debug_create_vertex_attribute_from_dict(self, attr_name, dict_id_to_val, default_val=np.nan):
        if (not hasattr(dict_id_to_val, 'items')) or (not hasattr(dict_id_to_val.items, '__call__')):
            raise ValueError('the method create_vertex_attribute_from_dict expects a dictionnary-like object, ' +
                             'which has a method \'items\'.')
        if not isinstance(attr_name, str):
            raise TypeError("the name of a vertex attribute should be a string.")
        for key, _ in dict_id_to_val.items():
            if key not in self.dict_cell_id_to_ind:
                raise ValueError(str(key) + ' is not the id of any vertex in the graph.')

    def create_vertex_attribute_from_dict(self, attr_name, dict_id_to_val, default_val):
        """
        Creates a new vertex attribute and populates its values using a dictionary-like object, whose keys are id of
        vertices, and values the corresponding attribute values. Note that you can specify a default value for the
        vertices not appearing in the dictionary.

        IMPORTANT: first, the method creates an array filled with the default value, and then replace the values in the
                   array using the dictionary. Therefore, the dtype of the attribute will be defined using the default
                   value. Thus, the user should either chose a default value with appropriate dtype, or change the
                   type of the attribute after creating the attribute.

        :param attr_name: string, name of the attribute.
        :param dict_id_to_val: Dictionary like object, whose keys are id of vertices, and values the corresponding
                               attribute value.
        :param default_val: Value used for the vertices for which an attribute value is not provided.
        """
        arr_attr = np.full((self.number_vertices,), default_val)
        for key, val in dict_id_to_val.items():
            arr_attr[self.dict_cell_id_to_ind[key]] = val
        self.df_attributes[attr_name] = arr_attr

    def change_type_attribute(self, attr_name, str_type):
        """
        Change the dtype of the selected attribute. Note that the type should be supported by DataFrameXS
        :param attr_name: string, name of the attribute
        :param str_type: string, target dtype of the attribute
        """
        self.df_attributes.change_type(attr_name, str_type)


class PeriodicAttributes:
    """
    Class that adds methods to define periodically varying arguments.
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_attributes'):
            self.df_attributes = DataFrameXS()

    def update_periodic_attribute(self, time, attr_name, amplitude, period, phase, intercept, condition=None):
        """
        Call this method to update the value of an attribute using the following formula.

            amplitude * np.sin(2 * math.pi * time / period + phase) + intercept

        Where time is either the value of the attribute 'time' of the graph, or if 'time' parameter is not None

        :param time: float or int, used as time parameter in the update formula.
        :param attr_name: string, name of the attribute
        :param amplitude: float, see formula above
        :param phase: float, see formula above
        :param period: float, see formula above
        :param intercept: float, see formula above
        :param condition: optional, default None. Boolean Array saying for which cell to apply the sinusoidal variation.
            If None, this method behave like an array of True has been provided.
        """
        arr_attr = self.df_attributes[attr_name]
        if condition is None:
            condition = np.full(arr_attr.shape, True, dtype=np.bool_)
        if time is None:
            time = self.time
        compute_sin_attr_with_condition(arr_attr, condition, time, amplitude,
                                        period, phase, intercept)


class AttributesFrom2DArraysSquareGrids:
    """
    Allow the user to add attributes based on 2D arrays. Designed to work with 'SquareGrids' topologies.
    """
    def __init__(self, **kwargs):
        pass

    def create_attribute_from_2d_array(self, attr_name, array_2d):
        """
        Create or update an attribute based on a 2D array input.

        :param attr_name: string, name of the attribute
        :param array_2d: 2d array
        """
        if array_2d.shape != self.shape:
            raise ValueError('Shapes do not match. Graph of shape ' + str(self.shape) +
                             ' while array of shape ' + str(array_2d.shape) + '.')
        arr_attr = np.full((self.number_vertices,), array_2d[0][0])
        for i in range(array_2d.shape[0]):
            for j in range(array_2d.shape[1]):
                arr_attr[self.dict_cell_id_to_ind[(i, j)]] = array_2d[i][j]

        self.df_attributes[attr_name] = arr_attr
