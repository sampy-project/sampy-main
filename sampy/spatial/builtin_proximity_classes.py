from .proximity_3d import (BaseProximity3dFromArrays, 
                           BaseProximity3dFromLatLonGrid,
                           Proximity3dBasicSpatialQueries)
from ..utils.decorators import sampy_class


@sampy_class
class Proximity3DFromLatLonGrid(BaseProximity3dFromLatLonGrid,
                                Proximity3dBasicSpatialQueries):
    """
    A proximity class is basically a wrapper around a cKDTree object. That is, a data structure
    optimized for some spatial queries. In SamPy, Proximity class main purpose is to find the 
    position of an agent with respect to some set of points. Those class also allow to define
    "allowed point" and, thus to know if an agent is on an allowed position or not.

    Here, the set of points consists in a grid on the earth defined with lat-lon coordinates.

    NOTE: This class initial purpose is to model arctic foxes movement on sea-ice, which is an 
          unstable surface. Which mean that, during a winter season, a given patch of sea ice can 
          switch between a "solid" and "melted" state multiple time. Which is the reason why we 
          introduced the notion of "allowed point", so that we can easilly update the status of each
          patch of ice.

    :param grid_lats: mandatory kwargs, 2d array of floats, latitudes of the points.
    :param grid_lons: mandatory kwargs, 2d array of floats, longitudes of the points.
    :param sphere_radius: mandatory kwarg, float, radius of the sphere on which the points live.
                          Mainly here to allow user to chose their unit of measure.
    :param arr_radius_point: optional, 1d array of float, default None. Gives the "radius" around 
                             each point of the proximity object. If not provided, it is computed
                             for each point of the object.
    :param allowed_points: optional, 2D array of bool, default None.
    """
    def __init__(self, grid_lats=None, grid_lons=None, sphere_radius=None, arr_radius_point=None, 
                 allowed_points=None, **kwargs):
        pass
    

@sampy_class
class Proximity3DFromArrays(BaseProximity3dFromArrays,
                            Proximity3dBasicSpatialQueries):
    """
    A proximity class is basically a wrapper around a cKDTree object. That is, a data structure
    optimized for some spatial queries. In SamPy, Proximity class main purpose is to find the 
    position of an agent with respect to some set of points. Those class also allow to define
    "allowed point" and, thus to know if an agent is on an allowed position or not.

    Here, the set of points is given explicitely by their 3D coordinates.

    :param arr_radius_point: mandatory kwarg, 1d array of float, default None. Gives the "radius" 
                             around each point of the proximity object. 
    :param coord_x: mandatory kwarg, 1D array of floats.
    :param coord_y: mandatory kwarg, 1D array of floats.
    :param coord_z: mandatory kwarg, 1D array of floats.
    :param allowed_points: optional, 2D array of bool, default None.
    """
    def __init__(self, arr_radius_point=None, coord_x=None, coord_y=None, coord_z=None, 
                 allowed_points=None, **kwargs):
        pass
