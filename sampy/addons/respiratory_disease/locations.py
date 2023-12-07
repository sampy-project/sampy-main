from ...pandas_xs.pandas_xs import DataFrameXS
import numpy as np
from .jit_compiled_functions import expand_array_according_to_condition


class Locations:
    """
    """
    def __init__(self, dict_type_loc_to_nb, rng_seed=1):
        self.rng = np.random.default_rng(seed=rng_seed)

        self.df_attributes = DataFrameXS()
        self.dict_type_loc_to_ind = dict()
        
        arr_type_loc = []
        counter = 0
        for type_loc, nb_loc in dict_type_loc_to_nb.items():
            self.dict_type_loc_to_ind[type_loc] = counter
            arr_type_loc = arr_type_loc + [counter for _ in range(nb_loc)]
            counter += 1
        self.df_attributes['type_loc'] = arr_type_loc

    @property
    def number_locations(self):
        return self.df_attributes.nb_rows

    def get_locs(self, type_loc):
        """
        Returns a 1D array of bool telling which location is of the requested type.

        :param type_loc: string, name of a location type. If 'all', returns an array full of True.
                         This last option is convenient for the internal of some methods of this
                         class.

        :returns: 1D array of bool, where returned_value[i] is True if and only if the location
                  of index i is of the requested type.
        """
        if type_loc.lower() == 'all':
            return np.full((self.number_locations,), True, dtype=np.bool_)
        try:
            return self.df_attributes['type_loc'] == self.dict_type_loc_to_ind[type_loc]
        except KeyError:
            raise ValueError("The parameter " + str(type_loc) + " is not the name of a location type.")
        
    def _sampy_debug_create_attribute(self, attr_name, value):
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

    def create_attribute(self, attr_name, value):
        """
        Creates a new location attribute and populates its values. Accepted input for 'value' are:
                - None: in this case, the attribute column is set empty
                - A single value, in which case all locations will have the same attribute value
                - A 1D array, which will become the attribute column.

        Note that if you use a 1D array, then you are implicitly working with the indexes of the locations, that is that
        the value at position 'i' in the array corresponds to the attribute value associated with the location whose index
        is 'i'.

        :param attr_name: string, name of the attribute
        :param value: either None, a single value, or a 1D array.
        """
        if self.df_attributes.nb_rows == 0 and len(np.array(value).shape) == 0:
            self.df_attributes[attr_name] = [value for _ in range(self.weights.shape[0])]
        else:
            self.df_attributes[attr_name] = value

    def add_random_attributes_from_sample(self, name, loc_type, arr_to_sample, 
                                          arr_sample_prob, default_val=-1):
        """
        Add a new attribute to the locations objects by doing some 
        """
        arr_loc_type = self.get_locs(loc_type)
        arr_sample = self.rng.choice(arr_to_sample, p=arr_sample_prob, 
                                       size=arr_loc_type.sum())
        self.df_attributes[name] = expand_array_according_to_condition(arr_sample, arr_loc_type,
                                                                       default_val)