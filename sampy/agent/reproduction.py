import numpy as np
from .jit_compiled_functions import (reproduction_find_random_mate_on_position,
                                     reproduction_find_random_mate_on_position_condition,
                                     reproduction_find_random_mate_on_position_polygamous,
                                     reproduction_find_random_mate_on_position_polygamous_condition)
from ..pandas_xs.pandas_xs import DataFrameXS

from .. utils.errors_shortcut import (check_input_is_permutation,
                                      check_input_array,
                                      check_col_exists_good_type,
                                      check_if_gender_array)

# ---------------------------------------------------------------------------------------------------------------------
# female encoded with 1, male with 0.
# ---------------------------------------------------------------------------------------------------------------------


class FindMateMonogamous:
    """
    This class provides methods for a monogamous agent to find mates
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()
        self.df_population['mom_id'] = None
        self.df_population['dad_id'] = None
        self.df_population['gender'] = None
        self.df_population['is_pregnant'] = None
        self.df_population['current_mate'] = None

        if not hasattr(self, 'dict_default_val'):
            self.dict_default_val = {}
        self.dict_default_val['mom_id'] = -1
        self.dict_default_val['dad_id'] = -1
        self.dict_default_val['gender'] = 1
        self.dict_default_val['is_pregnant'] = False
        self.dict_default_val['current_mate'] = -1

    def _sampy_debug_find_random_mate_on_position(self,
                                                  prob_get_pregnant,
                                                  shuffle=True,
                                                  permutation=None,
                                                  condition=None,
                                                  id_attribute='col_id',
                                                  position_attribute='position',
                                                  gender_attribute='gender',
                                                  mate_attribute='current_mate',
                                                  pregnancy_attribute='is_pregnant'):
        if self.df_population.nb_rows == 0:
            return

        if permutation is not None:
            check_input_is_permutation(permutation, 'permutation', self.df_population.nb_rows)

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

        check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute',
                                   prefix_dtype='int', reject_none=True)
        check_col_exists_good_type(self.df_population, mate_attribute, 'mate_attribute',
                                   prefix_dtype='int', reject_none=True)
        check_col_exists_good_type(self.df_population, pregnancy_attribute, 'pregnancy_attribute',
                                   prefix_dtype='bool', reject_none=True)

        check_if_gender_array(self.df_population[gender_attribute])

    def find_random_mate_on_position(self,
                                     prob_get_pregnant,
                                     shuffle=True,
                                     permutation=None,
                                     condition=None,
                                     id_attribute='col_id',
                                     position_attribute='position',
                                     gender_attribute='gender',
                                     mate_attribute='current_mate',
                                     pregnancy_attribute='is_pregnant'):
        """
        Find a mate on the current position of the agent. This mate is randomly picked. By default, the attribute used
        as the position if 'position', but the user may want to use 'territory' instead. For that purpose, the key-word
        argument 'position_attribute' can be used.

        :param prob_get_pregnant: float between 0 and 1. Probability that after mating the female will get pregnant.
        :param shuffle: optional, boolean, default True. By default, in this method the random choice of a mate is done
                        by shuffling the DataFrameXS 'df_population'. If set to False, the df is not shuffled, so that
                        the first male in a cell is paired with the first female in the cell (as they appear in df),
                        the second male with the second female, and so on until there is no male anymore (or no female).
        :param permutation: optional, default None, 1D array of integer. If not None and shuffle is True, this
                            permutation is used to shuffle df_population.
        :param condition: optional, array of bool, default None. Tells which agents should be included.
        :param id_attribute: optional, string, default 'col_id'. Id attribute of the agent. It is not recommended to
                             change this column, as this column is considered internal, and in the future this fact
                             could be used in other methods.
        :param position_attribute: optional, string, default 'position'. Position attribute of the agents. Should be
                                   integers corresponding to indexes of the vertices of the graph on which the agents
                                   live.
        :param gender_attribute: optional, string, default 'gender'.
        :param mate_attribute: optional, string, default 'current_mate'.
        :param pregnancy_attribute: optional, string, default 'is_pregnant'.
        """
        if self.df_population.nb_rows == 0:
            return

        if shuffle:
            self.df_population.scramble(permutation=permutation)

        if condition is None:
            rand = np.random.uniform(0, 1, ((self.df_population[gender_attribute] == 1).sum(),))
            reproduction_find_random_mate_on_position(self.df_population[mate_attribute],
                                                      self.df_population[pregnancy_attribute],
                                                      self.df_population[id_attribute],
                                                      self.df_population[position_attribute],
                                                      self.df_population[gender_attribute],
                                                      self.graph.connections.shape[0],
                                                      rand,
                                                      prob_get_pregnant)
        else:
            rand = np.random.uniform(0, 1, (((self.df_population[gender_attribute] == 1) & condition).sum(),))
            reproduction_find_random_mate_on_position_condition(self.df_population[mate_attribute],
                                                                self.df_population[pregnancy_attribute],
                                                                self.df_population[id_attribute],
                                                                self.df_population[position_attribute],
                                                                self.df_population[gender_attribute],
                                                                self.graph.connections.shape[0],
                                                                rand,
                                                                prob_get_pregnant,
                                                                condition)

    def get_females(self):
        """
        :return: 1D array of bool telling which agents are females.
        """
        return self.df_population['gender'] == 1

    def get_males(self):
        """
        :return: 1D array of bool telling which agents are males.
        """
        return self.df_population['gender'] == 0


class FindMatePolygamous:
    """
    This class provides methods for a polygamous agent to find mates
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()
        self.df_population['mom_id'] = None
        self.df_population['dad_id'] = None
        self.df_population['gender'] = None
        self.df_population['is_pregnant'] = None
        self.df_population['current_mate'] = None

        if not hasattr(self, 'dict_default_val'):
            self.dict_default_val = {}
        self.dict_default_val['mom_id'] = -1
        self.dict_default_val['dad_id'] = -1
        self.dict_default_val['gender'] = 1
        self.dict_default_val['is_pregnant'] = False
        self.dict_default_val['current_mate'] = -1

    def _sampy_debug_find_random_mate_on_position(self,
                                                  prob_get_pregnant,
                                                  condition=None,
                                                  id_attribute='col_id',
                                                  position_attribute='position',
                                                  gender_attribute='gender',
                                                  mate_attribute='current_mate',
                                                  pregnancy_attribute='is_pregnant'):
        if self.df_population.nb_rows == 0:
            return

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

        check_col_exists_good_type(self.df_population, position_attribute, prefix_dtype='int', reject_none=True)
        check_col_exists_good_type(self.df_population, mate_attribute, prefix_dtype='int', reject_none=True)
        check_col_exists_good_type(self.df_population, pregnancy_attribute, prefix_dtype='bool', reject_none=True)

        check_if_gender_array(self.df_population[gender_attribute])

    def find_random_mate_on_position(self,
                                     prob_get_pregnant,
                                     condition=None,
                                     id_attribute='col_id',
                                     position_attribute='position',
                                     gender_attribute='gender',
                                     mate_attribute='current_mate',
                                     pregnancy_attribute='is_pregnant'):
        """
        Find a mate on the current position of the agent. This mate is randomly picked. By default, the attribute used
        as the position if 'position', but the user may want to use 'territory' instead. For that purpose, the key-word
        argument 'position_attribute' can be used.

        :param prob_get_pregnant: float between 0 and 1. Probability that after mating the female will get pregnant.
        :param condition: optional, array of bool, default None. Tells which agents should be included.
        :param id_attribute: optional, string, default 'col_id'. Id attribute of the agent. It is not recommended to
                             change this column, as this column is considered internal, and in the future this fact
                             could be used in other methods.
        :param position_attribute: optional, string, default 'position'. Position attribute of the agents. Should be
                                   integers corresponding to indexes of the vertices of the graph on which the agents
                                   live.
        :param gender_attribute: optional, string, default 'gender'.
        :param mate_attribute: optional, string, default 'current_mate'.
        :param pregnancy_attribute: optional, string, default 'is_pregnant'.
        """
        if self.df_population.nb_rows == 0:
            return

        if condition is None:
            nb_females = (self.df_population[gender_attribute] == 1).sum()
            rand_preg = np.random.uniform(0, 1, (nb_females,))
            rand_mate = np.random.uniform(0, 1, (nb_females,))
            reproduction_find_random_mate_on_position_polygamous(self.df_population[id_attribute],
                                                                 self.df_population[position_attribute],
                                                                 self.df_population[gender_attribute],
                                                                 self.df_population[mate_attribute],
                                                                 self.df_population[pregnancy_attribute],
                                                                 self.graph.connections.shape[0],
                                                                 rand_preg, rand_mate, prob_get_pregnant)
        else:
            nb_females = ((self.df_population[gender_attribute] == 1) & condition).sum()
            rand_preg = np.random.uniform(0, 1, (nb_females,))
            rand_mate = np.random.uniform(0, 1, (nb_females,))
            reproduction_find_random_mate_on_position_polygamous_condition(self.df_population[id_attribute],
                                                                           self.df_population[position_attribute],
                                                                           self.df_population[gender_attribute],
                                                                           self.df_population[mate_attribute],
                                                                           self.df_population[pregnancy_attribute],
                                                                           self.graph.connections.shape[0],
                                                                           rand_preg, rand_mate, prob_get_pregnant,
                                                                           condition)

    def get_females(self):
        """
        :return: 1D array of bool telling which agents are females.
        """
        return self.df_population['gender'] == 1

    def get_males(self):
        """
        :return: 1D array of bool telling which agents are males.
        """
        return self.df_population['gender'] == 0


class OffspringCreationWithCustomProb:
    """
    This class sole purpose is to add the method create_offsprings_custom_prob. See its documentation below.
    """
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_create_offsprings_custom_prob(self,
                                                   arr_nb_children,
                                                   arr_prob_nb_children,
                                                   condition=None,
                                                   dico_default_values=None,
                                                   prob_failure=None,
                                                   age_attribute='age',
                                                   mother_attribute='mom_id',
                                                   father_attribute='dad_id',
                                                   gender_attribute='gender',
                                                   id_attribute='col_id',
                                                   position_attribute='position',
                                                   territory_attribute='territory',
                                                   mate_attribute='current_mate',
                                                   pregnancy_attribute='is_pregnant'):
        if self.df_population.nb_rows == 0:
            return

        check_input_array(arr_nb_children, 'arr_nb_children', 'int', nb_dim=1)
        check_input_array(arr_prob_nb_children, 'arr_prob_nb_children', 'float', nb_dim=1)

        if arr_nb_children.shape != arr_prob_nb_children.shape:
            raise ValueError("Arguments 'arr_nb_children' and 'arr_prob_nb_children' have different shapes.")

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

        if dico_default_values is not None:
            if not hasattr(dico_default_values, 'items') or hasattr(getattr(dico_default_values, 'items'), '__call__'):
                raise TypeError("The argument 'dico_default_value' should be a dictionnary-like object. Namely, have a "
                                "method called 'items' allowing to loop through keys and values.")

        check_col_exists_good_type(self.df_population, age_attribute, 'age_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, mother_attribute, 'mother_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, father_attribute, 'father_attribute', prefix_dtype='int',
                                   reject_none=True)

        check_if_gender_array(self.df_population[gender_attribute])

        check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, territory_attribute, 'territory_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, mate_attribute, 'mate_attribute', prefix_dtype='int',
                                   reject_none=True)

        check_col_exists_good_type(self.df_population, pregnancy_attribute, 'pregnancy_attribute', prefix_dtype='bool',
                                   reject_none=True)

    def create_offsprings_custom_prob(self,
                                      arr_nb_children,
                                      arr_prob_nb_children,
                                      condition=None,
                                      dico_default_values=None,
                                      prob_failure=None,
                                      age_attribute='age',
                                      mother_attribute='mom_id',
                                      father_attribute='dad_id',
                                      gender_attribute='gender',
                                      id_attribute='col_id',
                                      position_attribute='position',
                                      territory_attribute='territory',
                                      mate_attribute='current_mate',
                                      pregnancy_attribute='is_pregnant'):
        """
        Creates offsprings using two 1D arrays of same size, 'arr_nb_children' and 'arr_prob_nb_children', being
        respectively an array of integers and an array of non-negative floats, where for any index i,
        arr_prob_nb_children[i] is the probability for pregnant females to give birth to arr_nb_children[i]
        offsprings.

        Note that arr_prob_nb_children is normalized so that it sums to 1.

        :param arr_nb_children: 1D array of int, see above description.
        :param arr_prob_nb_children: 1d array of floats, see above description.
        :param condition: optional, 1d array of bool, default None. Tells which female and pregnant agents are allowed
                          to give birth.
        :param dico_default_values: optional, dictionnary, default None. Contains default values for the offsprings
                                    attributes. Any attribute not provided will use default values built-in the
                                    population object.
        :param prob_failure: optional, float, default None. Probability for an agent trying to give birth to fail such
                             action. As a result, the agent would no longer be pregnant and would not produce any
                             offspring.
        :param age_attribute: optional, string, default 'age'. Agent attribute used as age
        :param mother_attribute: optional, string, default 'mom_id'. Agent attribute used as mother id
        :param father_attribute: optional, string, default 'dad_id'. Agent attribute used as father id
        :param id_attribute: optional, string, default 'col_id'. Agent attribute used as agent id.
        :param position_attribute: optional, string, default 'position'.
        :param territory_attribute: optional, string, default 'territory'. If no territory in the model, the user should
                                    set it to None.
        :param mate_attribute: optional, string, default 'current_mate'. Agent attribute used as id of the mate.
        :param pregnancy_attribute: optional, string, default 'is_pregnant'.
        """
        if self.df_population.nb_rows == 0:
            return

        if dico_default_values is None:
            dico_default_values = dict()

        selected_females = self.df_population[pregnancy_attribute]
        if condition is not None:
            selected_females = selected_females & condition
        if prob_failure is not None:
            selected_females = selected_females & \
                               (np.random.uniform(0, 1, (self.df_population.nb_rows,)) >= prob_failure)

        df_selected_female = self.df_population[selected_females]

        if df_selected_female.nb_rows == 0:
            return

        # get number of babies per females
        prob = arr_prob_nb_children.astype('float64')
        prob = prob/prob.sum()
        arr_nb_baby = np.random.choice(arr_nb_children, df_selected_female.nb_rows, p=prob)

        arr_non_zero_babies = arr_nb_baby > 0
        df_selected_female = df_selected_female[arr_non_zero_babies]

        if df_selected_female.nb_rows == 0:
            return

        arr_nb_baby = arr_nb_baby[arr_non_zero_babies]

        # start building the children DataFrame
        df_children = DataFrameXS()
        df_children[mother_attribute] = np.repeat(df_selected_female[id_attribute], arr_nb_baby, axis=0)
        df_children[father_attribute] = np.repeat(df_selected_female[mate_attribute], arr_nb_baby, axis=0)
        df_children[position_attribute] = np.repeat(df_selected_female[position_attribute], arr_nb_baby, axis=0)
        if territory_attribute is not None:
            df_children[territory_attribute] = np.repeat(df_selected_female[territory_attribute], arr_nb_baby, axis=0)

        # defines the gender of the offsprings
        gender = 1 * (np.random.uniform(0, 1, (df_children.shape[0],)) >= 0.5)
        df_children[gender_attribute] = gender

        # fill non trivial attributes
        df_children[pregnancy_attribute] = False
        df_children[age_attribute] = 0
        df_children[id_attribute] = np.arange(self.counter_id, self.counter_id + df_children.shape[0])
        self.counter_id = self.counter_id + df_children.shape[0]

        # take care of the provided default values
        for attr, def_value in dico_default_values.items():
            df_children[attr] = def_value

        # take care of the rest
        set_treated_col = set([mother_attribute, father_attribute, position_attribute, territory_attribute,
                               gender_attribute, pregnancy_attribute, age_attribute, id_attribute])
        for col_name in self.df_population.list_col_name:
            if col_name in set_treated_col or col_name in dico_default_values:
                continue
            if col_name in self.dict_default_val:
                df_children[col_name] = self.dict_default_val[col_name]
            else:
                df_children[col_name] = None

        # set pregnancy of female that gave birth to False
        self.df_population[pregnancy_attribute] = self.df_population[pregnancy_attribute] & ~selected_females

        # concatenate the two dataframe
        self.df_population.concat(df_children)

