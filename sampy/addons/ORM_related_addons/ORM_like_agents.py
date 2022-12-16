from ...agent.base import BaseAgingAgent
from ...agent.mortality import NaturalMortalityOrmMethodology, OffspringDependantOnParents
from ...agent.reproduction import (OffspringCreationWithCustomProb,
                                      FindMateMonogamous,
                                      FindMatePolygamous)
from ...agent.movement import TerritorialMovementWithoutResistance
from ...utils.decorators import sampy_class

from .jit_compiled_function import *
from ...agent.jit_compiled_functions import movement_mov_around_territory_fill_bool_mov_using_condition

import numpy as np


class ComponentsFromORM:
    """
    This class contains basic methods extracted from ORM source code. Developed for the need of the Leighton Lab.
    Add orm-like methods to the agents. That are:

        - movement that take into account 'resistance' to the movement (i.e. each agent movement has a probability of
          success that depends on the user's landscape;
        - when an agent fails to move (for any reason), its displacement ends for the current timestep;
        - add a form of discrete correlated random walk, using the hexagonal structure of ORM landscapes (see
          dispersion);
        - add a series of ways to modelize natural mortality, as found in ORM code;
        - some methods make the assumption that each timestep represent a week, with 52 weeks in a year.

    IMPORTANT: for some methods, the underlying graph is assumed to come from an ORM xml using the class
               GraphFromORMxml
    """
    def __init__(self, **kwargs):
        self.df_population['has_moved'] = False
        self.dict_default_val['has_moved'] = False

    def orm_dispersion(self, timestep, permissible_weeks, condition, arr_nb_steps, arr_prob_nb_steps,
                       position_attribute='position', territory_attribute='territory', reflexion=False):
        """
        This method ignores resistance to movement.

        In the ORM, agents' dispersion is dealt with on a yearly basis. That is, each year (52 timesteps each
        representing a week) an agent will disperse once and only once. This method takes care of checking if an agent
        already dispersed or not. When an agent disperse, it draws a number of steps and performs a discrete
        correlated random walk on the hexagonal landscape. At first, the agent choses a random direction among the six
        possible neighbours of its current cell (North, north-east, south-est, etc. See GraphORMXml description for
        details). If there is no cell in that direction, dispersion stops (but it is still counted as if the agent
        dispersed this year). If there is one, the agent jumps to this new cell, then a new direction is picked: with a
        60% chance the direction remains unchanged, with 20% it turns left by one notch (if for instance the first
        direction was North-east, then there is a 20% chance that the next direction will be North), with 20% it turns
        right. We then repeat the process until the agent has made all its steps.

        IMPORTANT: This method assumes agents are living on a GraphFromORMxml.
        Warning: If reflexion is used, there might be a bias since we recycle an already used random value.

        :param timestep: current timestep, number of week since simulation started (starts at 0).
        :param permissible_weeks: array-like object of integer saying which week of the year the agents can disperse
                                  (counted from 0 to 51).
        :param condition: 1D array of bool of shape (nb_agents,) saying which agents are allowed to disperse.
        :param arr_nb_steps: 1D array of non-negative integers, works in tandem with arr_prob_nb_steps.
        :param arr_prob_nb_steps: 1D array of float, each between 0 and 1. arr_prob_nb_steps[i] is the probability for
                                  a dispersing agent to do arr_nb_steps[i] steps. Note that this number of step can be
                                  0. Effectively, this would lead to the agent "having dispersed this year" yet without
                                  having moved.
        :param position_attribute: optional, string, default 'position'. Name of the position attribute in the agents.
        :param territory_attribute: optional, string, default 'territory'. Name of the territory attribute in the
                                    agents.
        :param reflexion: optionnal, boolean, default False.
        """
        if self.df_population.nb_rows == 0:
            return

        # we reinitialize the 'has_moved' status if first week of the year
        if timestep % 52 == 0:
            self.df_population['has_moved'] = False

        if timestep % 52 not in permissible_weeks:
            return

        can_move = condition & ~self.df_population['has_moved']
        will_move = np.random.uniform(0, 1, can_move.sum()) < \
                    (permissible_weeks.index(timestep % 52) + 1) / len(permissible_weeks)
        prob = arr_prob_nb_steps.astype('float64')
        prob = prob / prob.sum()
        rand_nb_steps = np.random.choice(arr_nb_steps, will_move.sum(), p=prob)
        rand_directions = np.random.uniform(0, 1, rand_nb_steps.sum())

        if reflexion:
            orm_like_agent_dispersion_with_reflexion(can_move, will_move, rand_nb_steps, rand_directions,
                                                     self.df_population[position_attribute],
                                                     self.df_population[territory_attribute],
                                                     self.df_population['has_moved'],
                                                     self.graph.connections, self.graph.weights)
        else:
            orm_like_agent_orm_dispersion(can_move, will_move, rand_nb_steps, rand_directions,
                                          self.df_population[position_attribute],
                                          self.df_population[territory_attribute],
                                          self.df_population['has_moved'],
                                          self.graph.connections)

    def orm_dispersion_with_resistance(self, timestep, permissible_weeks, condition, arr_nb_steps, arr_prob_nb_steps,
                                       position_attribute='position', territory_attribute='territory'):
        """
        This method included resistance to movement.

        In the ORM, agents' dispersion is dealt with on a yearly basis. That is, each year (52 timesteps each
        representing a week) an agent will disperse once and only once. This method takes care of checking if an agent
        already dispersed or not. When an agent disperse, it draws a number of steps and performs a discrete
        correlated random walk on the hexagonal landscape. At first, the agent choses a random direction among the six
        possible neighbours of its current cell (North, north-east, south-est, etc. See GraphORMXml description for
        details). If there is no cell in that direction, dispersion stops (but it is still counted as if the agent
        dispersed this year). If there is one, the agent jumps to this new cell, then a new direction is picked: with a
        60% chance the direction remains unchanged, with 20% it turns left by one notch (if for instance the first
        direction was North-east, then there is a 20% chance that the next direction will be North), with 20% it turns
        right. We then repeat the process until the agent has made all its steps.

        IMPORTANT: This method assumes agents are living on a GraphFromORMxml.
        Warning: If reflexion is used, there might be a bias since we recycle an already used random value.

        :param timestep: current timestep, number of week since simulation started (starts at 0).
        :param permissible_weeks: array-like object of integer saying which week of the year the agents can disperse
                                  (counted from 0 to 51).
        :param condition: 1D array of bool of shape (nb_agents,) saying which agents are allowed to disperse.
        :param arr_nb_steps: 1D array of non-negative integers, works in tandem with arr_prob_nb_steps.
        :param arr_prob_nb_steps: 1D array of float, each between 0 and 1. arr_prob_nb_steps[i] is the probability for
                                  a dispersing agent to do arr_nb_steps[i] steps. Note that this number of step can be
                                  0. Effectively, this would lead to the agent "having dispersed this year" yet without
                                  having moved.
        :param position_attribute: optional, string, default 'position'. Name of the position attribute in the agents.
        :param territory_attribute: optional, string, default 'territory'. Name of the territory attribute in the
                                    agents.
        """
        if self.df_population.nb_rows == 0:
            return

        # we reinitialize the 'has_moved' status if first week of the year
        if timestep % 52 == 0:
            self.df_population['has_moved'] = False

        if timestep % 52 not in permissible_weeks:
            return

        can_move = condition & ~self.df_population['has_moved']
        will_move = np.random.uniform(0, 1, can_move.sum()) < \
                    (permissible_weeks.index(timestep % 52) + 1) / len(permissible_weeks)
        prob = arr_prob_nb_steps.astype('float64')
        prob = prob / prob.sum()
        rand_nb_steps = np.random.choice(arr_nb_steps, will_move.sum(), p=prob)
        rand_directions = np.random.uniform(0, 1, rand_nb_steps.sum())
        rand_res = np.random.uniform(0, 1, rand_nb_steps.sum())

        orm_like_agent_orm_dispersion_with_resistance(can_move, will_move, rand_nb_steps, rand_directions,
                                                      self.df_population[position_attribute],
                                                      self.df_population[territory_attribute],
                                                      self.df_population['has_moved'], self.graph.connections,
                                                      self.graph.prob_successful_move,
                                                      rand_res)

    def mov_around_with_resistance(self,
                                   proba_remain_on_territory,
                                   condition=None,
                                   territory_attribute='territory',
                                   position_attribute='position'):
        """
        This method includes movement resistance. Update the average position of the agent around its territory during
        the current time step.

        :param proba_remain_on_territory: float, probability to stay on the territory
        :param condition: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent (i.e. the agent at the line i of df_population) can
                          move.
        :param territory_attribute: optional, string, default 'territory'
        :param position_attribute: optional, string, default 'position'
        """
        if self.df_population.nb_rows == 0:
            return

        if condition is not None:
            pre_bool_mov = np.random.uniform(0, 1, condition.sum()) > proba_remain_on_territory
            bool_mov = movement_mov_around_territory_fill_bool_mov_using_condition(pre_bool_mov, condition)
        else:
            bool_mov = np.random.uniform(0, 1, self.df_population.shape[0]) > proba_remain_on_territory

        rand_direction = np.random.uniform(0, 1, bool_mov.sum())
        rand_res = np.random.uniform(0, 1, bool_mov.sum())

        orm_like_agents_mov_around_with_resistance(self.df_population[territory_attribute],
                                                   self.df_population[position_attribute],
                                                   self.graph.connections, self.graph.weights,
                                                   self.graph.prob_successful_move, bool_mov, rand_direction,
                                                   rand_res)

    def _mortality_from_v08(self, arr_annual_mortality, condition_count, alpha_beta=None, condition=None,
                            shuffle=True, age_attribute='age', position_attribute='position'):
        """
        (made by Francois Viard)

        This is an adaptation of the mortality method found in ARM v08 (file cFox.cs). ARM is not a typo, it is the
        'Arctic Rabies Model', which is a variation of ORM. I don't think it's been published or used in publication?

        WARNING: There are no sources given in the ORM file for the origin of this method. As such, I cannot guarantee
                 that I correctly understood what it is supposed to do and why. Therefore, this method is considered
                 private and should be use with caution.

        :param arr_annual_mortality: array of float between 0 and 1. arr_annual_mortality[i] is the target probability
                                     for an agent to die at age i.
        :param condition_count: 1D array of bool, tells which agent to count.
        :param alpha_beta: Absolute mystery. In the formula used in ARM there is an Alpha and a Beta, but I could not
                           figure out if those were actually modified or if the default values were always taken. I
                           finally decided to postpone the inclusion of those option to a later date... However,
                           every project finally moved away from this method and fell back to the usual ORM mortality.
                           As it is, DO NOT USE THIS KWARGS.
        :param condition: optional, 1D array of bool, default None. If not None, tells which agents can die.
        :param shuffle: optional, bool, default True. If True, shuffle the population dataframe before applying method
        :param age_attribute: optional, string, default 'age'. Name of the age attribute of the agents.
        :param position_attribute: optional, string, default 'position'. Name of the position attribute of the agents.
        """
        if self.df_population.nb_rows == 0:
            return

        if shuffle:
            permutation = self.df_population.scramble(return_permutation=True)
            if condition is not None:
                condition = condition[permutation]
            condition_count = condition_count[permutation]

        count_arr = self.count_pop_per_vertex(position_attribute=position_attribute, condition=condition_count)
        if alpha_beta is None:
            if condition is None:
                rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                survive = orm_like_agents_mortality_from_v08_no_condition_no_alpha_beta(count_arr, condition_count,
                                                                                        self.df_population[position_attribute],
                                                                                        self.graph.df_attributes['K'],
                                                                                        arr_annual_mortality,
                                                                                        self.df_population[age_attribute],
                                                                                        rand)
            else:
                rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                survive = orm_like_agents_mortality_from_v08_with_condition_no_alpha_beta(count_arr, condition_count,
                                                                                          self.df_population[
                                                                                            position_attribute],
                                                                                          self.graph.df_attributes['K'],
                                                                                          arr_annual_mortality,
                                                                                          self.df_population[
                                                                                            age_attribute],
                                                                                          rand, condition)
        self.df_population = self.df_population[survive]

    def _mortality_from_v08_with_gender(self, arr_female_annual_mortality, arr_male_annual_mortality, condition_count,
                                        alpha_beta=None, condition=None, shuffle=True, age_attribute='age',
                                        position_attribute='position', gender_attribute='gender'):
        """
        (made by Francois Viard)

        This is an adaptation of the mortality method found in ARM v08 (file cFox.cs). ARM is not a typo, it is the
        'Arctic Rabies Model', which is a variation of ORM. I don't think it's been published or used in publication?

        WARNING: There are no sources given in the ORM file for the origin of this method. As such, I cannot guarantee
                 that I correctly understood what it is supposed to do and why. Therefore, this method is considered
                 private and should be use with caution.

        :param arr_female_annual_mortality: array of float between 0 and 1. arr_annual_mortality[i] is the target
                                    probability for a female agent to die at age i.
        :param arr_male_annual_mortality: array of float between 0 and 1. arr_annual_mortality[i] is the target
                                    probability for a male agent to die at age i.
        :param condition_count: 1D array of bool, tells which agent to count.
        :param alpha_beta: Absolute mystery. In the formula used in ARM there is an Alpha and a Beta, but I could not
                           figure out if those were actually modified or if the default values were always taken. I
                           finally decided to postpone the inclusion of those option to a later date... However,
                           every project finally moved away from this method and fell back to the usual ORM mortality.
                           As it is, DO NOT USE THIS KWARGS.
        :param condition: optional, 1D array of bool, default None. If not None, tells which agents can die.
        :param shuffle: optional, bool, default True. If True, shuffle the population dataframe before applying method
        :param age_attribute: optional, string, default 'age'. Name of the age attribute of the agents.
        :param position_attribute: optional, string, default 'position'. Name of the position attribute of the agents.
        :param gender_attribute: optional, string, default 'gender'. Name of the gender attribute of the agents.
        """
        if self.df_population.nb_rows == 0:
            return

        if shuffle:
            permutation = self.df_population.scramble(return_permutation=True)
            if condition is not None:
                condition = condition[permutation]
            condition_count = condition_count[permutation]

        count_arr = self.count_pop_per_vertex(position_attribute=position_attribute, condition=condition_count)
        if alpha_beta is None:
            if condition is None:
                rand = np.random.uniform(0, 1, self.df_population.nb_rows)
                survive = orm_like_agents_mortality_from_v08_with_gender_no_condition_no_alpha_beta(count_arr,
                                                                                condition_count,
                                                                                self.df_population[position_attribute],
                                                                                self.graph.df_attributes['K'],
                                                                                self.df_population[gender_attribute],
                                                                                arr_female_annual_mortality,
                                                                                arr_male_annual_mortality,
                                                                                self.df_population[age_attribute],
                                                                                rand)
            else:
                rand = np.random.uniform(0, 1, (condition.sum(),))
                survive = orm_like_agents_mortality_from_v08_with_gender_with_condition_no_alpha_beta(count_arr,
                                                                                condition_count,
                                                                                self.df_population[position_attribute],
                                                                                self.graph.df_attributes['K'],
                                                                                self.df_population[gender_attribute],
                                                                                arr_female_annual_mortality,
                                                                                arr_male_annual_mortality,
                                                                                self.df_population[age_attribute],
                                                                                rand,
                                                                                condition)
        self.df_population = self.df_population[survive]


class ExperimentalDensityDependentMortality:
    """
    Some experiments that will be moved elsewhere soon.
    """
    def __init__(self, **kwargs):
        self._corrected_beta_female = None
        self._corrected_beta_male = None

    def compute_and_save_beta_female(self, alpha, arr_female_weekly_mortality):
        self._corrected_beta_female = 1. + (1. / alpha) * np.log((1. / arr_female_weekly_mortality) - 1.)

    def compute_and_save_beta_male(self, alpha, arr_male_weekly_mortality):
        self._corrected_beta_male = 1. + (1. / alpha) * np.log((1. / arr_male_weekly_mortality) - 1.)

    def experimental_mortality_logistic_function(self, alpha, arr_female_weekly_mortality, arr_male_weekly_mortality,
                                                 age_attribute='age', position_attribute='position', k_attribute='K',
                                                 shuffle=True, gender_attribute='gender'):
        if self._corrected_beta_female is None:
            self.compute_and_save_beta_female(alpha, arr_female_weekly_mortality)
        if self._corrected_beta_male is None:
            self.compute_and_save_beta_male(alpha, arr_male_weekly_mortality)
        if shuffle:
            self.df_population.scramble()
        count_arr = self.count_pop_per_vertex(position_attribute=position_attribute)
        rand = np.random.uniform(0, 1, (self.df_population.nb_rows,))
        survive = experimental_density_mortality(count_arr, self.df_population[position_attribute],
                                                 self.df_population[age_attribute],
                                                 self.graph.df_attributes[k_attribute],
                                                 self.df_population[gender_attribute],
                                                 self._corrected_beta_male, self._corrected_beta_female,
                                                 alpha, rand)
        self.df_population = self.df_population[survive]


@sampy_class
class ORMLikeAgent(BaseAgingAgent,
                   NaturalMortalityOrmMethodology,
                   OffspringDependantOnParents,
                   FindMateMonogamous,
                   OffspringCreationWithCustomProb,
                   TerritorialMovementWithoutResistance,
                   ComponentsFromORM):
    """
    Basic ORM like agents.

    Includes main components from SamPy plus the building block 'ComponentsFromORM'.

    :param graph: Mandatory kwargs, a graph object. Some methods expect this to be a "GraphFromORMXml' object.
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class ORMMongooses(BaseAgingAgent,
                   NaturalMortalityOrmMethodology,
                   OffspringDependantOnParents,
                   FindMatePolygamous,
                   OffspringCreationWithCustomProb,
                   TerritorialMovementWithoutResistance,
                   ComponentsFromORM):
    """
    Mongooses from ORM with some updates for Caroline Sauve projects.
    Includes main components from SamPy plus the building block 'ComponentsFromORM'.

    :param graph: Mandatory kwargs, a graph object. Some methods expect this to be a GraphFromORMXml object.
    :param pregnancy_duration: mandatory kwargs, an integer. Duration in weeks of pregnancy.
    """
    def __init__(self, pregnancy_duration=None, **kwargs):
        if pregnancy_duration is None:
            raise ValueError("A value for pregnancy duration should be given using kwarg 'pregnancy_duration'")
        self.pregnancy_duration = pregnancy_duration
        self.df_population['nb_weeks_being_pregnant'] = 0
        self.dict_default_val['nb_weeks_being_pregnant'] = 0

        self.df_population['week_next_potential_mating'] = -1
        self.dict_default_val['week_next_potential_mating'] = -1

    def increment_number_of_weeks_of_pregnancy(self):
        """
        Mongoose object keeps track of pregnancy duration with individual counters. This method increments those
        counters. Should be called weekly.
        """
        self.df_population['nb_weeks_being_pregnant'] += self.df_population['is_pregnant']

    def check_if_mother_free_of_dependent_young(self, age_independence, min_age_reproduction, mother_attribute='mom_id',
                                                id_attribute='col_id', age_attribute='age'):
        """
        Checks which females old enough to reproduce do not have offsprings.

        :param age_independence: integer, age at which offspring can live on their own.
        :param min_age_reproduction: integer, age at which females can reproduce.
        :param mother_attribute: optional, string, default 'mom_id'. Name of the attribute giving the id of the mother
                                 of each agent.
        :param id_attribute: optional, string, default 'col_id'. Name of the attribute containing the id of the agents.
        :param age_attribute: optional, string, default 'age'. Name of the attribute containing the age of the agents.

        :return: 1D array of bool, where return[i] is True if and only if the i-th agent is a Female available for
                 reproduction.
        """
        offsprings = self.df_population[age_attribute] < age_independence
        mom_id_offsprings = self.df_population[mother_attribute][offsprings]
        potential_mothers = (self.df_population[age_attribute] >= min_age_reproduction) & self.get_females()
        orm_mongooses_check_if_mother_free_of_juveniles(mom_id_offsprings, self.df_population[id_attribute],
                                                        potential_mothers)
        return potential_mothers

    def give_birth_if_needed(self, arr_nb_children, arr_prob_nb_children, condition=None, prob_failure=None):
        """
        Make each female that reached the end of their pregnancy period give birth.

        :param arr_nb_children: 1D array of integer, works in tandem with arr_prob_nb_children below.
        :param arr_prob_nb_children: 1D array of float between 0 and 1. arr_prob_nb_children[i] is the probability for
                                     a female giving birth to have arr_nb_children[i] offsprings.
        :param condition: optional, 1D array of bool, default None. If not None, tells which female agents can give
                          birth.
        :param prob_failure: optional, float between 0 and 1, default one. If not None, gives the probability for the
                             agent to fail giving birth. Note that this is equivalent with putting 0 as a possible value
                             in arr_nb_children, and we recommend user to chose one solution or the other.
        """
        about_to_give_birth = (self.df_population['nb_weeks_being_pregnant'] >= self.pregnancy_duration)
        if condition is None:
            condition = about_to_give_birth
        else:
            condition = condition & about_to_give_birth
        self.df_population['nb_weeks_being_pregnant'] *= ~condition
        self.create_offsprings_custom_prob(arr_nb_children, arr_prob_nb_children, condition=condition,
                                           prob_failure=prob_failure)

    def children_follow_their_mom(self, min_age, max_age, age_attribute='age', mom_id_attribute='mom_id',
                                  id_attribute='col_id', position_attribute='position',
                                  territory_attribute='territory'):
        youngs = (self.df_population[age_attribute] >= min_age) & (self.df_population[age_attribute] <= max_age)
        id_moms = self.df_population[mom_id_attribute][youngs]
        orm_mongooses_update_ter_pos_youngs(id_moms, youngs, self.df_population[territory_attribute],
                                            self.df_population[position_attribute], self.df_population[id_attribute],
                                            self.df_population[mom_id_attribute])

    def weekly_mating_checks_and_update(self, current_week, mean_mate_weeks, var_mate_weeks, age_independence,
                                        min_age_reproduction, gender_attribute='gender', mother_attribute='mom_id',
                                        id_attribute='col_id', age_attribute='age', position_attribute='position',
                                        pregnancy_attribute='is_pregnant', mate_attribute='current_mate'):
        """
        Method written in order to reproduce the multiple reproduction spikes that can be seen in Mongooses.

        :param current_week: integer, value between 0 and 51 representing the current week of the year.
        :param mean_mate_weeks:
        :param var_mate_weeks:
        :param age_independence:
        :param min_age_reproduction:
        :param gender_attribute:
        :param mother_attribute:
        :param id_attribute:
        :param age_attribute:
        :param position_attribute:
        :param pregnancy_attribute:
        :param mate_attribute:
        """

        # we begin with making the females ready to mate attempt to, well, mate
        potentially_mating_females = self.df_population['week_next_potential_mating'] == current_week
        if potentially_mating_females.sum() > 0:
            female_available = self.check_if_mother_free_of_dependent_young(age_independence, min_age_reproduction,
                                                                            gender_attribute=gender_attribute,
                                                                            mother_attribute=mother_attribute,
                                                                            id_attribute=id_attribute,
                                                                            age_attribute=age_attribute)
            female_available = female_available & ~self.df_population[pregnancy_attribute]
            mating_females = potentially_mating_females & female_available
            mating_males = self.get_males() & (self.df_population[age_attribute] >= min_age_reproduction)
            self.find_random_mate_on_position(1., condition=(mating_females | mating_males), id_attribute=id_attribute,
                                              position_attribute=position_attribute, gender_attribute=gender_attribute,
                                              mate_attribute=mate_attribute, pregnancy_attribute=pregnancy_attribute)

        # we now (re)-initialize the "week_next_potential_mating" attribute
        init_next_week = potentially_mating_females | ((self.df_population['week_next_potential_mating'] == -1) &
                                                       (self.df_population[age_attribute] >= min_age_reproduction))
        nb_init_to_perform = init_next_week.sum()
        if nb_init_to_perform > 0:
            rand_gauss = np.random.normal(0, 1, (nb_init_to_perform,))
            orm_mongooses_update_mating_week(current_week, mean_mate_weeks, var_mate_weeks,
                                             self.df_population['week_next_potential_mating'],
                                             init_next_week, rand_gauss)



