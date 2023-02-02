from .base import BaseTwoSpeciesDisease
from .transition import TransitionCustomProbPermanentImmunity
from .transmission import ContactTransmissionSameGraph
from ...utils.decorators import sampy_class


@sampy_class
class TwoSpeciesContactCustomProbTransitionPermanentImmunity(BaseTwoSpeciesDisease,
                                                             TransitionCustomProbPermanentImmunity,
                                                             ContactTransmissionSameGraph):
    """
    Basic disease, transmission by direct contact (contagion only between agents on the same vertex), transition between
    disease states encoded by user given arrays of probabilities, and permanent immunity.

    IMPORTANT: We strongly recommend the user to use the "simplified" methods defined here instead of the usual
               'contaminate_vertices', 'contact_contagion' and 'transition_between_states'. Indeed, the combination of
               building blocks involved in this disease requires many actions to be performed in a precise order,
               otherwise the model's behaviour cannot be guaranteed. See each simplified method description to learn
               about each respective ordering.

    :param disease_name: mandatory kwargs. String.
    :param host1: mandatory kwargs. Population object of the first host.
    :param host2: mandatory kwargs. Population object of the second host.
    """
    def __init__(self, **kwargs):
        pass

    def simplified_contact_contagion(self, contact_rate_matrix, arr_timesteps_host1, arr_prob_timesteps_host1,
                                     arr_timesteps_host2, arr_prob_timesteps_host2,
                                     position_attribute_host1='position', position_attribute_host2='position',
                                     condition_host1=None, condition_host2=None,
                                     return_arr_new_infected=False, return_type_transmission=False):
        """
        Propagate the disease by direct contact using the following methodology. For any vertex X of the graph, we
        count the number of contagious agents N_c_host_j, then each non immuned agent of host_k on the vertex X has a
        probability of

                            1 - (1 - contact_rate_matrix[j,k]) ** N_c_host_j

        to become infected by a contact with an agent of host_j.

        Detailed Explanation: each agent has a series of counter attached, telling how much time-steps they will spend
                              in each disease status. Those counters have to be initialized when an individual is newly
                              infected, and that's what this method does to the newly infected individuals.

        :param contact_rate_matrix: 2D array of floats of shape (2, 2). Here, contact_rate_matrix[0][0] is the
                                    probability of contact contagion from host1 to host1, contact_rate_matrix[0][1] is
                                    the probability of contact contagion from host1 to host2, etc...
        :param arr_timesteps_host1: 1d array of int. work in tandem with arr_prob_timesteps_host1, see below.
        :param arr_prob_timesteps_host1: 1D array of float. arr_prob[i] is the probability for an agent to stay infected
                                         but not contagious for arr_nb_timestep[i] time-steps.
        :param arr_timesteps_host2: same but for host2.
        :param arr_prob_timesteps_host2: same but for host2.
        :param position_attribute_host1: optional, string, default 'position'. Name of the agent attribute used as
                                         position for host1.
        :param position_attribute_host2: optional, string, default 'position'. Name of the agent attribute used as
                                         position for host2.
        :param condition_host1: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent of host1 (i.e. the agent at the line i of df_population)
                          can be infected and transmit disease. All the agent having their corresponding value to False
                          are protected from infection and cannot transmit the disease.
        :param condition_host2: optional, array of bool, default None. Array of boolean such that the i-th value is
                          True if and only if the i-th agent of host2 (i.e. the agent at the line i of df_population)
                          can be infected and transmit disease. All the agent having their corresponding value to False
                          are protected from infection and cannot transmit the disease.
        :param return_arr_new_infected: optional, boolean, default False. If True, the method returns two arrays telling
                                        which agent got contaminated in each host species.
        :param return_type_transmission: optional, boolean, default False.

        :return: Depending on the values of the parameters 'return_arr_new_infected' and 'return_type_transmission',
                 the return value will either be None (both are False) or a dictionnary whose key-values are:
                    - 'arr_new_infected_host1', 'arr_new_infected_host2' if return_arr_new_infected is True, and the
                      values are 1D arrays of bool telling which agents got infected for each host;
                    - 'arr_type_transmission_host1', 'arr_type_transmission_host2' if return_type_transmission is True,
                      and the values are 1D arrays of non-negative integers such that the integer at line i is 0 if the
                      i-th agent has not been contaminated, 1 if it has been contaminated by a member of its own species
                      2 if it has been contaminated by a member of the other species, 3 if it has been contaminated by
                      agents from both its species and the other.
        """
        dict_contagion = self.contact_contagion(contact_rate_matrix, return_arr_new_infected=True,
                                                return_type_transmission=return_type_transmission,
                                                position_attribute_host1=position_attribute_host1,
                                                position_attribute_host2=position_attribute_host2,
                                                condition_host1=condition_host1, condition_host2=condition_host2)
        self.initialize_counters_of_newly_infected('host1', dict_contagion['arr_new_infected_host1'],
                                                   arr_timesteps_host1, arr_prob_timesteps_host1)
        self.initialize_counters_of_newly_infected('host2', dict_contagion['arr_new_infected_host2'],
                                                   arr_timesteps_host2, arr_prob_timesteps_host2)
        if not return_type_transmission and not return_arr_new_infected:
            return
        if return_arr_new_infected:
            return dict_contagion
        else:  # no other case needed
            del dict_contagion['arr_new_infected_host1']
            del dict_contagion['arr_new_infected_host2']
            return dict_contagion

    def simplified_transition_between_states(self, prob_death_host1, prob_death_host2,
                                             arr_infectious_period_host1, arr_prob_infectious_period_host1,
                                             arr_infectious_period_host2, arr_prob_infectious_period_host2):
        """
        Takes care of the transition between all the disease states. That is, agents that are at the end of their
        infected period become contagious and agents at the end of their contagious period either die (with a
        probability of 'prob_death') or become immuned.

        Detailed Explanation: the method transition_between_states is coded in such a way that when using it for
                              transitionning from con to imm, all the agents at the end of their contagious period at
                              the time the method is called transition. Therefore, we have to make the transition
                              'con' to 'death' first.

        :param prob_death_host1: float between 0 and 1, probability for an agent of host1 to die at the end of the
                                 contagious period
        :param prob_death_host2: float between 0 and 1, probability for an agent of host2 to die at the end of the
                                 contagious period
        :param arr_infectious_period_host1: 1d array of int, works in tandem with arr_prob_infectious_period_host1.
                                            See Below.
        :param arr_prob_infectious_period_host1: 1d array of floats, sums to 1. Same shape as
                    arr_infectious_period_host1. When an agent transition from infected to contagious, then
                    arr_prob_infectious_period_host1[i] is the probability for this agent to stay
                    arr_infectious_period_host1[i] timesteps contagious.
        :param arr_infectious_period_host2: same as host1
        :param arr_prob_infectious_period_host2: same as host1
        """
        self.transition_between_states('host1', 'con', 'death', proba_death=prob_death_host1)
        self.transition_between_states('host2', 'con', 'death', proba_death=prob_death_host2)

        if self.host1.df_population.nb_rows != 0:
            self.transition_between_states('host1', 'con', 'imm')
            self.transition_between_states('host1', 'inf', 'con', arr_nb_timestep=arr_infectious_period_host1,
                                           arr_prob_nb_timestep=arr_prob_infectious_period_host1)

        if self.host2.df_population.nb_rows != 0:
            self.transition_between_states('host2', 'con', 'imm')
            self.transition_between_states('host2', 'inf', 'con', arr_nb_timestep=arr_infectious_period_host2,
                                           arr_prob_nb_timestep=arr_prob_infectious_period_host2)

    def simplified_contaminate_vertices(self, host, list_vertices, level, arr_timesteps, arr_prob_timesteps,
                                        condition=None, position_attribute='position',
                                        return_arr_newly_contaminated=True):
        """
        Contaminate a list of vertices.

        Detailed explanation: each agent has a series of counter attached, telling how much time-steps they will spend
                              in each disease status. Those counters have to be initialized when an individual is newly
                              infected, and that's what this method does to the newly infected individuals.

        :param host: string, either 'host1' or 'host2', tells which host to infect.
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

        :return: if return_arr_newly_contaminated is set to True, returns a 1D array of bool telling which agents where
                 contaminated. Returns None otherwise.
        """
        arr_new_contaminated = self.contaminate_vertices(host, list_vertices, level,
                                                         condition=condition, position_attribute=position_attribute,
                                                         return_arr_newly_contaminated=True)
        self.initialize_counters_of_newly_infected(host, arr_new_contaminated, arr_timesteps, arr_prob_timesteps)
        if return_arr_newly_contaminated:
            return arr_new_contaminated
