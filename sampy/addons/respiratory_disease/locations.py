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
        Add a new attribute to the locations objects by doing some sampling in an array of values.
        This method is probabilistic.

        :param name: string, name of the new attribute.
        :param loc_type: string, name of the location type. Use 'all' if every location should have
                         have a value for the attribute.
        :param arr_to_sample: array or list. Values to sample from for the new attribute.
        :param arr_sample_prob: array or list of float, sum to 1. Probability for each value in
                                arr_to_sample to be sampled as an attribute of a location.
        :param default_val: optional, default -1. Values to assign by default to any location that
                            is not of the right location type. 
        """
        arr_loc_type = self.get_locs(loc_type)
        arr_sample = self.rng.choice(arr_to_sample, p=arr_sample_prob, 
                                       size=arr_loc_type.sum())
        self.df_attributes[name] = expand_array_according_to_condition(arr_sample, arr_loc_type,
                                                                       default_val)
        
    def add_attributes_from_proportion(self, name, loc_type, arr_values, arr_proportion,
                                       default_val=-1):
        """
        Add a new attribute to the locations objects associating a value from the parameter
        arr_values according to the proportions given in arr_proportion.
        This method is deterministic.

        IMPORTANT: This method does some rounding, and there might be some small variations on the
                   resulting proportions. Most notably, the last values given in arr_values will be
                   used to fill any remaining empty attribute within the selected location type.

        :param name: string, name of the new attribute.
        :param loc_type: string, name of the location type. Use 'all' if every location should have
                         have a value for the attribute.
        :param arr_values: array or list. Values to sample from for the new attribute.
        :param arr_proportion: array or list of float, sum to 1. Proportion of each value in
                                arr_to_sample in the resulting assignation. The fact that this 
                                arrays sum to 1 is important. If it is not the case, the resulting
                                attribute distribution may not follow at all the expected 
                                proportions.
        :param default_val: optional, default -1. Values to assign by default to any location that
                            is not of the right location type. 
        """
        arr_loc_type = self.get_locs(loc_type)
        nb_loc = arr_loc_type.sum()

        list_new_attributes = []
        nb_loc_with_attribute = 0
        for i, prop in enumerate(arr_proportion):
            nb_loc_current_prop = int(np.floor(prop * nb_loc))
            nb_loc_with_attribute += nb_loc_current_prop
            if nb_loc_with_attribute > nb_loc:
                nb_remaining_loc = nb_loc - len(list_new_attributes)
                list_new_attributes.extend([arr_values[i] for _ in range(nb_remaining_loc)])
                break
            else:
                list_new_attributes.extend([arr_values[i] for _ in range(nb_loc_current_prop)])

        if len(list_new_attributes) < nb_loc:
            nb_remaining_loc = nb_loc - len(list_new_attributes)
            list_new_attributes.extend([arr_values[-1] for _ in range(nb_remaining_loc)])

        if len(list_new_attributes) > nb_loc:
            raise ValueError("The resulting array of attribute is too long. This is likely a code problem.")

        arr_attributes = np.array(list_new_attributes)
        self.df_attributes[name] = expand_array_according_to_condition(arr_attributes,
                                                                       arr_loc_type, default_val)
