import numpy as np
from .jit_compiled_functions import *
from ..pandas_xs.pandas_xs import DataFrameXS
from ..utils.errors_shortcut import (check_col_exists_good_type,
                                     check_input_array,
                                     check_input_is_permutation,
                                     check_if_gender_array)


class NaturalMortalityOrmMethodology:
    """
    This class provides methods to modelize natural mortality.
    See description of method 'natural_death_orm_methodology' for a precise description.
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'df_population'):
            self.df_population = DataFrameXS()

    def _sampy_debug_natural_death_orm_methodology(self, array_death_proba_male, array_death_proba_female,
                                                   shuffle=False,
                                                   permutation_shuffle=None,
                                                   condition=None,
                                                   condition_count=None,
                                                   gender_attribute='gender',
                                                   age_attribute='age',
                                                   position_attribute='position',
                                                   k_factor_attribute='K'):
        if self.df_population.nb_rows == 0:
            return

        check_col_exists_good_type(self.df_population, position_attribute, 'position_attribute', 'int',
                                   reject_none=True)
        check_col_exists_good_type(self.graph.df_attributes, k_factor_attribute, 'k_factor_attribute', 'float',
                                   reject_none=True)

        # we check that the 2 first arrays are 1D float array
        check_input_array(array_death_proba_female, 'array_death_proba_female', 'float', nb_dim=1)
        check_input_array(array_death_proba_male, 'array_death_proba_male', 'float', nb_dim=1)

        # we need to check that there is no animal whose age is bigger than the size of the prob_arrays.
        # in order to do so, we need to check that age_attribute and gender_attribute are properly used.
        check_col_exists_good_type(self.df_population, age_attribute, 'age_attribute',
                                   prefix_dtype='int', reject_none=True)

        # now the gender attribute
        check_col_exists_good_type(self.df_population, gender_attribute, 'gender_attribute',
                                   prefix_dtype='int', reject_none=True)
        check_if_gender_array(self.df_population[gender_attribute])

        # now we check that there are no animal too old.
        max_female = self.df_population[age_attribute][self.df_population[gender_attribute] == 1].max()
        if max_female >= array_death_proba_female.shape[0]:
            raise ValueError("There is a female agent of age " + str(max_female) + " while the array "
                             "array_death_proba_female covers only ages until " +
                             str(array_death_proba_female.shape[0] - 1) + ".")

        max_male = self.df_population[age_attribute][self.df_population[gender_attribute] == 0].max()
        if max_male >= array_death_proba_male.shape[0]:
            raise ValueError("There is a male agent of age " + str(max_male) + " while the array "
                             "array_death_proba_male covers only ages until " +
                             str(array_death_proba_male.shape[0] - 1) + ".")

        if permutation_shuffle is not None:
            if not shuffle:
                raise ValueError("A value for permutation_shuffle has been provided while shuffle is deactivated.")
            check_input_is_permutation(permutation_shuffle, 'permutation_shuffle', length=self.df_population.nb_rows)

        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

        if condition_count is not None:
            check_input_array(condition_count, 'condition_count', 'bool', shape=(self.df_population.nb_rows,))

    def natural_death_orm_methodology(self, array_death_proba_male, array_death_proba_female,
                                      bias=0.,
                                      shuffle=True,
                                      permutation_shuffle=None,
                                      condition=None,
                                      condition_count=None,
                                      gender_attribute='gender',
                                      age_attribute='age',
                                      position_attribute='position',
                                      k_factor_attribute='K',
                                      return_death_dataframe=False
                                      ):
        """
        Kill agents using ORM methodology. I.E. the probability for a given agent to die is given by the formula:

                            'prob_death * ( bias + (nb_agents_on_cell / K_cell) )'

        Where prob_death is a parameter depending on the age of the agent, nb_agents_on_cell is the number of agent on
        the cell of the tested agent (this number is updated along the process), and K is the K parameter of the current
        cell. This method takes the gender into account.

        The user can shuffle the dataframe before applying this function using the kwarg 'shuffle', and supply a
        permutation to be used for the shuffling using the kwarg 'permutation_shuffle'. The user can also
        specify two conditions as 1D boolean arrays:

            - condition: array telling which agents are susceptible to be killed
            - condition_count: array telling which agents are counted to determine the number of agents in each cell.
                               This is useful if, for instance, the user wants to exclude juveniles from the
                               nb_agents_on_cell used in the above formula.

        WARNING: if shuffle is used, then any data previously extracted using the row position of the agents in the
                 DataFrame df_population will be irrelevant.

        :param array_death_proba_male: 1D array of float where array_death_proba_male[i] is the proba of death of a
                                       male agent whose age is 'i'.
        :param array_death_proba_female: 1D array of float where array_death_proba_female[i] is the proba of death of a
                                         female agent whose age is 'i'.
        :param bias: optional, float, default 0. Hyperparameter used to configure the mortality. Represents a 'base
                     mortality'.
        :param shuffle: optional, boolean, default True.
        :param permutation_shuffle: optional, 1D array of int, default None. Permutation used for the shuffle. If
                                    shuffle is activated and no permutation is provided, then one is generated using
                                    numpy.random.permutation
        :param condition: optional, 1D array of bool, default None. Array telling which agent are susceptible to die.
        :param condition_count: optional, 1D array of bool, default None. Array telling which agent are to be counted.
        :param gender_attribute: optional, string, default 'gender'. Name of the column containing gender info.
        :param age_attribute: optional, string, default 'age'. Name of the column containing age info.
        :param position_attribute: optional, string, default 'position'. Name of the column containing position info.
        :param k_factor_attribute: optional, string, default 'K'. Name of the column containing K value in graph
                                   df_attribute DataFrame.
        :param return_death_dataframe: optional, boolean, default False. If True, returns a DataFrameXS containing all
                                       the agents killed during this method call

        :return: if return_death_dataframe is set to True, returns a DataFrameXS containing the agents killed during
                 this method call.
        """
        if self.df_population.nb_rows == 0:
            return
        # shuffle the population dataframe if required
        if shuffle:
            used_permutation = self.df_population.scramble(permutation=permutation_shuffle, return_permutation=True)
            if condition is not None:
                condition = condition[used_permutation]
            if condition_count is not None:
                condition_count = condition_count[used_permutation]

        # count agents per cell
        count_arr = self.count_pop_per_vertex(position_attribute=position_attribute, condition=condition_count)
        if condition is None:
            if condition_count is None:
                arr_survive = mortality_natural_death_orm_methodology(bias,
                                                                      np.random.uniform(0, 1,
                                                                                        self.df_population.nb_rows),
                                                                      array_death_proba_male,
                                                                      array_death_proba_female, count_arr,
                                                                      self.df_population[position_attribute],
                                                                      self.graph.df_attributes[k_factor_attribute],
                                                                      self.df_population[age_attribute],
                                                                      self.df_population[gender_attribute])
            else:
                arr_survive = mortality_natural_death_orm_methodology_condition_count(
                                                                bias,
                                                                np.random.uniform(0, 1, self.df_population.nb_rows),
                                                                array_death_proba_male,
                                                                array_death_proba_female, count_arr,
                                                                condition_count,
                                                                self.df_population[position_attribute],
                                                                self.graph.df_attributes[k_factor_attribute],
                                                                self.df_population[age_attribute],
                                                                self.df_population[gender_attribute])
        else:
            if condition_count is None:
                arr_survive = mortality_natural_death_orm_methodology_condition_death(
                                                                bias,
                                                                np.random.uniform(0, 1, self.df_population.nb_rows),
                                                                array_death_proba_male,
                                                                array_death_proba_female, count_arr,
                                                                condition,
                                                                self.df_population[position_attribute],
                                                                self.graph.df_attributes[k_factor_attribute],
                                                                self.df_population[age_attribute],
                                                                self.df_population[gender_attribute])
            else:
                arr_survive = mortality_natural_death_orm_methodology_both_cond(
                                                                bias,
                                                                np.random.uniform(0, 1, self.df_population.nb_rows),
                                                                array_death_proba_male,
                                                                array_death_proba_female, count_arr,
                                                                condition, condition_count,
                                                                self.df_population[position_attribute],
                                                                self.graph.df_attributes[k_factor_attribute],
                                                                self.df_population[age_attribute],
                                                                self.df_population[gender_attribute])

        if return_death_dataframe:
            death_df = self.df_population[~arr_survive]
            self.df_population = self.df_population[arr_survive]
            return death_df
        else:
            self.df_population = self.df_population[arr_survive]

    def kill_too_old(self, limit_age, age_attribute='age'):
        """
        Kill agents whose age is greater or equal to the specified limit age.

        :param limit_age: integer. Any agent older than this parameter will be killed.
        :param age_attribute: optional, string, default 'age'. Name of the age attribute in the population dataframe.
        """
        self.df_population = self.df_population[self.df_population[age_attribute] < limit_age]


class OffspringDependantOnParents:
    """
    This class add the possibility to kill the dependent offsprings whose mother is dead.
    """
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_kill_children_whose_mother_is_dead(self, age_limit,
                                                        id_attribute='col_id',
                                                        age_attribute='age',
                                                        mother_id_attribute='mom_id'):
        if self.df_population.nb_rows == 0:
            return
        check_col_exists_good_type(self.df_population, age_attribute, 'age_attribute', prefix_dtype='int',
                                   reject_none=True)
        check_col_exists_good_type(self.df_population, mother_id_attribute, 'mother_id_attribute', prefix_dtype='int',
                                   reject_none=True)

    def kill_children_whose_mother_is_dead(self, age_limit,
                                           id_attribute='col_id',
                                           age_attribute='age',
                                           mother_id_attribute='mom_id'):
        """
        Kill the agents that are not independent yet and whose mother is dead. Note that an agent whose age is precisely
        age_limit is not considered independent.

        :param age_limit: integer. Age of independence.
        :param id_attribute: optional, string, default 'col_id'
        :param age_attribute: optional, string, default 'age'
        :param mother_id_attribute: optional, string, default 'mom_id'
        """
        if self.df_population.nb_rows == 0:
            return
        young = self.df_population[age_attribute] <= age_limit
        has_mom = self.df_population.check_arr_in_col(self.df_population[mother_id_attribute], id_attribute,
                                                      condition=young)
        too_old = np.array(self.df_population[age_attribute] > age_limit)
        self.df_population = self.df_population[has_mom | too_old]


class KillPercentagePop:
    """
    This class adds the possibility to kill a given percentage of the population.
    """
    def __init__(self, **kwargs):
        pass

    def _sampy_debug_kill_proportion_of_population(self, proportion, condition=None):
        if condition is not None:
            check_input_array(condition, 'condition', 'bool', shape=(self.df_population.nb_rows,))

    def kill_proportion_of_population(self, proportion, condition=None):
        """
        For each agent, a random uniform number between 0 and 1 is sampled, and each agent for which this number is
        smaller than the argument 'proportion' is killed. Note that the user can use the kwarg 'condition' to
        exclude some agents from this test.

        :param proportion: float between 0 and 1.
        :param condition: optional, 1D array of bool, default None.
        """
        rand = np.random.uniform(0, 1, (self.df_population.nb_rows,)) > proportion
        if condition is not None:
            rand = rand & condition
        self.df_population = self.df_population[rand]
