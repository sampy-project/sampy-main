import numpy as np

# radius_earth=6371.008,


def extract_deviation_angles(list_lat, list_lon, input_in_radians=False):
    """
    Extract deviation angles from a sequence of positions given as lists of
    lattitude and longitude. Those deviation angles are computed in the 
    context of Spherical Correlated Random Walks (SCRW), as introduced in
    [paper in prep.].

    :param list_lat: list of float, lattitude of the positions. By default,
                     it is assumed to be given in degrees, but this behaviour
                     can be changed with the kwarg input_in_radians. Should be
                     of length at least 3.
    :param list_lon: list of float, longitudes of the positions. By default,
                     it is assumed to be given in degrees, but this behaviour
                     can be changed with the kwarg input_in_radians. Should be
                     of length at least 3.
    :param input_in_radians: optional, boolean, default False. If True, the
                             positions are assumed to be given in radians, and
                             in degrees otherwise.

    :returns: a 1D array of floats of shape (len(list_lat) - 2,)
    """
    # first we convert inputs to np array and check that they makes sense
    lats = np.array(list_lat)
    lons = np.array(list_lon)
    if lats.shape != lons.shape:
        raise ValueError("Parameters list_lat and list_lon should have same length.")
    if len(lats.shape) != 1:
        raise ValueError("Parameters list_lat and list_lon should be 1D lists or arrays.")
    if lats.shape[0] < 3:
        raise ValueError("At least three positions are needed to compute a deviation angle.")
    
    # we convert the inputs into radians if needed
    if not input_in_radians:
        lats = np.radians(lats)
        lons = np.radians(lons)

    # we turn those lat lon into 3D coordinates (on a sphere of unit radius, since it
    # does not impact the deviation angles).
    arr_x = np.cos(np.radians(lats)) * np.cos(np.radians(lons))
    arr_y = np.cos(np.radians(lats)) * np.sin(np.radians(lons))
    arr_z = np.sin(np.radians(lats))

    # we now compute the deviation angles
    for i in range(arr_x.shape[0]):
        pass