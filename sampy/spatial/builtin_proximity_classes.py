from .proximity_3d import (BaseProximity3dFromArrays, 
                           BaseProximity3dFromLatLonGrid,
                           Proximity3dBasicSpatialQueries)
from ..utils.decorators import sampy_class
import numpy as np


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
    
    def get_closest_points_from_lat_lon(self, arr_target_lat, arr_target_lon, nb_points=1):
        """
        Given a series of target-point whose coordinates are given as three 1 dimensional arrays, this method gives the
        index of the closest point (and the distance) in the proximity class to each of the target-points.

        :param arr_target_lat: 1d array of float, lat coordinate of target point.
        :param arr_target_lon: 1d array of float, lon coordinate of target point.
        :param nb_points: optional, integer, default 1. Number of closest points to the target.

        :return: couple of arrays (distance, indices), the second one giving the index of the closest point and the
            first one the distance to this point
        """
        arr_target_x = self.sphere_radius * np.cos(np.radians(arr_target_lat)) * np.cos(np.radians(arr_target_lon))
        arr_target_y = self.sphere_radius * np.cos(np.radians(arr_target_lat)) * np.sin(np.radians(arr_target_lon))
        arr_target_z = self.sphere_radius * np.sin(np.radians(arr_target_lat))
        stacked_arr = np.column_stack([arr_target_x, arr_target_y, arr_target_z])
        return self.kdtree.query(stacked_arr, k=nb_points)
    

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
