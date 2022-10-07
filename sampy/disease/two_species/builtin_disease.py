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
    """
    def __init__(self, **kwargs):
        pass

    def simplified_contact_contagion(self):
        pass

    def simplified_transition_between_states(self):
        pass

    def simplified_contaminate_vertices(self):
        pass
