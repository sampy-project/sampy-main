from ...pandas_xs.pandas_xs import DataFrameXS
from .jit_compiled_functions import allocate_homes_to_agents_basic
import numpy as np


class Agents:
    """
    
    """
    def __init__(self, number_agents, locations, age_weights, rng_seed=1):
        # here, we will follow the new numpy guidelines for RNG generation by creating a rng gen
        self.rng = np.random.default_rng(seed=rng_seed)

        # main dataframe
        self.df_population = DataFrameXS()

        # link to the locations
        self.locations = locations

        # define ids for each agent
        self.df_population['col_id'] = np.arange(number_agents, dtype=int)

        # give a random age to each agent using age weights
        self.df_population['age'] = self.rng.choice(np.arange(1, 101), p=age_weights, 
                                                    size=number_agents)
        
    @property
    def number_agents(self):
        return self.df_population.nb_rows
    
    def reset_rng(self, rng_seed):
        self.rng = np.random.default_rng(seed=rng_seed)
        
    def allocate_homes(self, type_loc, capacity_attribute='capacity', home_attribute='home',
                       nb_occupant_per_home_attribute='number_occupants'):
        """
        This method allocates homes to each agent, according to a location attribute that encodes
        the capacity of each selected location. This is based entire

        IMPORTANT: This method is moved out of the __init__ since it is very likely to be changed
                   / improved in the future. Therefore it will be easier to adapt new scripts once
                   other more realistic options are chosen.

        :param type_loc: string, type of the location to be used as home.
        :param capacity_attribute: optional, string, default 'capacity'. Attribute of the locations
                                   used as capacity of each home, i.e. their maximal number of 
                                   inhabitants.
        :param home_attribute: optional, string, default 'home'. Name of the attribute in which 
                               the home of each agent is stored.
        """
        arr_bool_homes = self.locations.get_locs(type_loc)
        capacity_homes = self.locations.df_attributes[capacity_attribute]
        if capacity_homes[arr_bool_homes].sum() > self.number_agents:
            raise ValueError("There are more agents than capacity in homes.")

        arr_homes_indexes = np.where(arr_bool_homes) # this array will be torned appart by the 
                                                     # following function call. DO NOT USE AFTERWARD
        arr_occupants_in_home = np.zeros(capacity_homes.shape, dtype=int)
        rand_unif = self.rng.uniform(0., 1., self.number_agents)
        home_array = allocate_homes_to_agents_basic(rand_unif, capacity_homes, self.number_agents,
                                                    arr_homes_indexes, arr_occupants_in_home)
        
        self.df_population[home_attribute] = home_array
        self.locations.create_attribute(nb_occupant_per_home_attribute, arr_occupants_in_home)

    def allocate_community(self, arr_age_groups, arr_prob_per_age_group, list_loc_type):
        """
        Allocate a community location to each agent. This is done randomly using age-group based
        probabilities.
        """
        pass