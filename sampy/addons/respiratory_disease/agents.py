from ...pandas_xs.pandas_xs import DataFrameXS
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
        
    def allocate_homes(self, type_loc, capacity_attribute='capacity', home_attribute='home'):
        """
        This method allocates homes to each agent, according to a location attribute that encodes
        the capacity of each selected location.

        :param type_loc: string, type of the location to be used as home.
        :param capacity_attribute: optional, string, default 'capacity'. Attribute of the locations
                                   used as capacity of each home, i.e. their maximal number of 
                                   inhabitants.
        :param home_attribute: optional, string, default 'home'. Name of the attribute in which 
                               the home of each agent is stored.
        """
        pass
