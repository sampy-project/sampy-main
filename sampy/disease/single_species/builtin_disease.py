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
    """
    def __init__(self, **kwargs):
        pass
