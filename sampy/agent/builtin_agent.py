from .base import BaseAgingAgent
from .mortality import NaturalMortalityOrmMethodology, OffspringDependantOnParents
from .reproduction import (FindMateMonogamous, 
                           FindMatePolygamous, 
                           OffspringCreationWithCustomProb,
                           ReproductionMonogamousWithMarker)
from .movement import TerritorialMovementWithoutResistance
from .random_walk import SphericalRandomWalk
from ..utils.decorators import sampy_class


@sampy_class
class BasicMammal(BaseAgingAgent,
                  NaturalMortalityOrmMethodology,
                  OffspringDependantOnParents,
                  FindMateMonogamous,
                  OffspringCreationWithCustomProb,
                  TerritorialMovementWithoutResistance):
    """
    Agent that represents a basic territorial mammal. This agent is Monogamous.
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class BasicMammalPolygamous(BaseAgingAgent,
                            NaturalMortalityOrmMethodology,
                            OffspringDependantOnParents,
                            FindMatePolygamous,
                            OffspringCreationWithCustomProb,
                            TerritorialMovementWithoutResistance):
    """
    Agent that represents a basic territorial mammal. This agent is Polygamous (in the sense that a given male can
    mate with several females during a single call of the method 'find_random_mate_on_position').
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class BasicMammalWithSCRW(BaseAgingAgent,
                          NaturalMortalityOrmMethodology,
                          OffspringDependantOnParents,
                          FindMateMonogamous,
                          OffspringCreationWithCustomProb,
                          TerritorialMovementWithoutResistance,
                          SphericalRandomWalk):
    """
    Agent that represents a basic territorial mammal able to perform spherical correlated random walk. 
    This agent is Monogamous.

    IMPORTANT: The theory behind SCRW is given in a paper in preparation. If it is not out yet, send
               a mail to francois.viard@umontreal.ca to get the current draft of the paper (all
               the maths are already included in the draft).
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class BasicMammalWithMarkersAndSCRW(BaseAgingAgent,
                                    NaturalMortalityOrmMethodology,
                                    OffspringDependantOnParents,
                                    ReproductionMonogamousWithMarker,
                                    TerritorialMovementWithoutResistance,
                                    SphericalRandomWalk):
    """
    Agent that represents a basic territorial mammal able to perform spherical correlated random walk. 
    This agent is Monogamous. This agent has markers.

    IMPORTANT: The theory behind SCRW is given in a paper in preparation. If it is not out yet, send
               a mail to francois.viard@umontreal.ca to get the current draft of the paper (all
               the maths are already included in the draft).
    """
    def __init__(self, **kwargs):
        pass





















# class TerritorialMammalWithRandomWalkOnSphere(BaseAgingAgent,
#                                               NaturalMortality,
#                                               OffspringDependantOnParents,
#                                               MammalReproduction,
#                                               TerritorialAnimal,
#                                               RandomWalkOnSphere):
#
#     def __init__(self, graph):
#         if hasattr(self, 'df_population'):
#             self.df_population['territory'] = None
#             self.df_population['position'] = None
#             self.df_population['mom_id'] = None
#             self.df_population['dad_id'] = None
#             self.df_population['gender'] = None
#             self.df_population['is_pregnant'] = None
#             self.df_population['current_mate'] = None
#             self.df_population['natural_mortality'] = None
#         else:
#             self.df_population = pd.DataFrame(columns=['territory', 'position', 'mom_id', 'dad_id', 'gender',
#                                                        'is_pregnant', 'current_mate', 'natural_mortality'])
#         super().__init__(graph)
#
#
# class TerritorialMammalWithRandomWalkOnSphereGenMarkers(BaseAgingAgent,
#                                                         NaturalMortality,
#                                                         OffspringDependantOnParents,
#                                                         MammalReproductionGenetikMarker,
#                                                         TerritorialAnimal,
#                                                         RandomWalkOnSphere):
#
#     def __init__(self, graph):
#         if hasattr(self, 'df_population'):
#             self.df_population['territory'] = None
#             self.df_population['position'] = None
#             self.df_population['mom_id'] = None
#             self.df_population['dad_id'] = None
#             self.df_population['gender'] = None
#             self.df_population['is_pregnant'] = None
#             self.df_population['current_mate'] = None
#             self.df_population['natural_mortality'] = None
#         else:
#             self.df_population = pd.DataFrame(columns=['territory', 'position', 'mom_id', 'dad_id', 'gender',
#                                                        'is_pregnant', 'current_mate', 'natural_mortality'])
#         super().__init__(graph)
