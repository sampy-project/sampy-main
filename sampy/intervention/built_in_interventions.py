from .vaccination import (BaseVaccinationSingleSpeciesDisease,
                          VaccinationSingleSpeciesDiseaseFixedDuration)
from .culling import (BaseCullingSingleSpecies,
                      CullingSingleSpecies)
from ..utils.decorators import sampy_class


@sampy_class
class BasicVaccination(BaseVaccinationSingleSpeciesDisease,
                       VaccinationSingleSpeciesDiseaseFixedDuration):
    """
    Provide basic vaccination capacities. Vaccines have a fixed duration, always the same for every vaccinated agent.
    When vaccinated, an agent is totally immuned to the disease.
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class BasicCulling(BaseCullingSingleSpecies,
                   CullingSingleSpecies):
    """
    Provide way to cull the agent population, notably by allowing to kill a given proportion of agents on cells.
    """
    def __init__(self, **kwargs):
        pass
