from .base import BaseSingleSpeciesDisease
from .transition import (TransitionCustomProbPermanentImmunity)
from .transmission import TransmissionByContact
from ...utils.decorators import sampy_class


# todo: the name of this class is way too long. We need to find a short and expressive name
@sampy_class
class ContactCustomProbTransitionPermanentImmunity(BaseSingleSpeciesDisease,
                                                   TransitionCustomProbPermanentImmunity,
                                                   TransmissionByContact):
    """
    Basic disease, transmission by direct contact (contagion only between agents on the same vertex), transition between
    disease states encoded by user given arrays of probabilities, and permanent immunity.

    IMPORTANT: We strongly recommend the user to use the "simplified" methods defined here instead of the usual
               'contaminate_vertices', 'contact_contagion' and 'transition_between_states'. Indeed, the combination of
               building blocks involved in this disease requires many actions to be performed in a precise order,
               otherwise the model's behaviour cannot be guaranteed. See each simplified method description to learn
               about each respective ordering.
    """
    def __init__(self, **kwargs):
        pass

    def simplified_contaminate_vertices(self, list_vertices, level, arr_timesteps, arr_prob_timesteps,
                                        condition=None, position_attribute='position',
                                        return_arr_newly_contaminated=False):
        """
        Contaminate a list of vertices.

        Detailed explanation: each agent has a series of counter attached, telling how much time-steps they will spend
                              in each disease status. Those counters have to be initialized when an individual is newly
                              infected, and that's what this method does to the newly infected individuals.

        :param list_vertices: list of vertices ID to be contaminated.
        :param level: float, probability for agent on the vertices to be contaminated.
        :param arr_timesteps: 1D array of integer. Works in tandem with 'arr_prob_timesteps'. See below.
        :param arr_prob_timesteps: 1D array of float. arr_prob_timesteps[i] is the probability for an agent to stay
                                   infected but not contagious for arr_timesteps[i] timesteps.
        :param condition: optional, array of bool, default None. If not None, say which agents are susceptible to be
                          contaminated.
        :param position_attribute: optional, string, default 'position'. Agent attribute to be used to define
                                   their position.
        :param return_arr_newly_contaminated: optional, boolean, default True. If True, the method returns an array
                                              telling which agents were contaminated.

        :return: if return_arr_newly_contaminated is set to True, returns a 1D array of bool. Returns None ortherwise.
        """
        #
        arr_new_contaminated = self.contaminate_vertices(list_vertices, level, return_arr_newly_contaminated=True,
                                                         condition=condition, position_attribute=position_attribute)
        self.initialize_counters_of_newly_infected(arr_new_contaminated, arr_timesteps, arr_prob_timesteps)
        if return_arr_newly_contaminated:
            return arr_new_contaminated

    def simplified_contact_contagion(self, contact_rate, arr_timesteps, arr_prob_timesteps,
                                     position_attribute='position', condition=None,
                                     return_arr_newly_contaminated=False):
        """
        Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
        count the number of contagious agents N_c, then each non immuned agent on the vertex X has a probability of

                            1 - (1 - contact_rate) ** N_c

        to become infected.

        Detailed Explanation: each agent has a series of counter attached, telling how much time-steps they will spend
                              in each disease status. Those counters have to be initialized when an individual is newly
                              infected, and that's what this method does to the newly infected individuals.

        :param contact_rate:
        :param arr_timesteps: 1D array of integer. Works in tandem with 'arr_prob_timesteps'. See below.
        :param arr_prob_timesteps: 1D array of float. arr_prob_timesteps[i] is the probability for an agent to stay
                                   infected but not contagious for arr_timesteps[i] timesteps.
        :param condition: optional, array of bool, default None. If not None, say which agents are susceptible to be
                          contaminated.
        :param position_attribute: optional, string, default 'position'. Agent attribute to be used to define
                                   their position.
        :param return_arr_newly_contaminated: optional, boolean, default True. If True, the method returns an array
                                              telling which agents were contaminated.

        :return: if return_arr_newly_contaminated is set to True, returns a 1D array of bool. Returns None ortherwise.
        """
        arr_new_contaminated = self.contact_contagion(contact_rate, position_attribute=position_attribute,
                                                      condition=condition, return_arr_new_infected=True)
        self.initialize_counters_of_newly_infected(arr_new_contaminated, arr_timesteps, arr_prob_timesteps)
        if return_arr_newly_contaminated:
            return arr_new_contaminated

    def simplified_transition_between_states(self, prob_death, arr_infectious_period, arr_prob_infectious_period):
        """
        Takes care of modifying

        :return:
        """
        self.transition_between_states('con', 'death', proba_death=prob_death)

        if self.host.df_population.nb_rows == 0:
            return

        self.transition_between_states('con', 'imm')
        self.transition_between_states('inf', 'con', arr_nb_timestep=arr_infectious_period,
                                       arr_prob_nb_timestep=arr_prob_infectious_period)
