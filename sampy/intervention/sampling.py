import numpy as np


class BaseSamplingSingleSpecies:
    """
    Base Class for single species sampling.
    """
    def __init__(self, target_species=None, **kwargs):
        if target_species is None:
            raise ValueError("No agent object provided for the sampling. Should be provided using kwarg "
                             "'target_species'.")
        self.target_species = target_species


class SamplingSingleSpecies:
    """
    Introduce the methods for sampling a single species
    """
    def __init__(self, **kwargs):
        pass

    def sample_proportion_from_array(self, array_proportion, condition=None, position_attribute='position',
                                     return_as_pandas_df=False, eliminate_sampled_pop=False):
        """
        Take as input an array telling the proportion of agent to sample in each vertex.

        :param array_proportion: 1D array of float. array_proportion[i] is the probability for an agent living in the
                                 vertex of index i.
        :param condition: optional, 1D array of bool, default None. If not None, tell which agent can be sampled.
        :param position_attribute: optional, string, default 'position'. Tell which attribute of the agents should
                                   be used as position.
        :param return_as_pandas_df: optional, boolean, default False. Clear.
        :param eliminate_sampled_pop: optional, boolean, default False. If True,

        :return: a DataFrameXS if return_as_pandas_df is False, a pandas dataframe otherwise. The returned DF is
                 the sample of the population taken from df_population
        """
        rand = np.random.uniform(0, 1, (self.target_species.df_population.nb_rows,))
