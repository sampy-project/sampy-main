from .proximity_3d import (BaseProximity3dFromArrays, 
                           BaseProximity3dFromLatLonGrid,
                           Proximity3dBasicSpatialQueries)
from ..utils.decorators import sampy_class


@sampy_class
class Proximity3DFromLatLonGrid(BaseProximity3dFromLatLonGrid,
                                Proximity3dBasicSpatialQueries):
    """
    :param grid_lats: mandatory kwargs, 2d array of floats, latitudes of the points.
    :param grid_lons: mandatory kwargs, 2d array of floats, longitudes of the points.
    :param sphere_radius: mandatory kwarg, float, radius of the sphere on which the points live.
    :param arr_radius_point: optional, 1d array of float, default None. Gives the "radius" around 
                             each point of the proximity object. If not provided, it is computed
                             for each point of the object.
    :param allowed_points: optional, 2D array of bool, default None.
    """
    def __init__(self, grid_lats=None, grid_lons=None, sphere_radius=None, arr_radius_point=None, allowed_points=None, **kwargs):
        pass