import numpy as np

# radius_earth=6371.008,


def extract_deviation_angles(list_lat, list_lon, input_in_radians=False):
    """
    Extract deviation angles from a sequence of positions given as lists of
    lattitude and longitude. Those deviation angles are computed in the 
    context of Spherical Correlated Random Walks (SCRW), as introduced in
    [paper in prep.].

    IMPORTANT: This function will have unpredictable behaviour if there are 
               two consecutive positions corresponding to antipodal points.
               Indeed, in this case any direction from the first position
               is admissible.

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

    # we swap to float64 to improve precision in the following computations
    arr_x = arr_x.astype(np.float64)
    arr_y = arr_y.astype(np.float64)
    arr_z = arr_z.astype(np.float64)

    # we now compute the deviation angles
    deviation_angles = []
    for i in range(arr_x.shape[0] - 2):
        start_pos =  np.array([arr_x[i],     arr_y[i],     arr_z[i]])
        middle_pos = np.array([arr_x[i + 1], arr_y[i + 1], arr_z[i + 1]])
        end_pos =    np.array([arr_x[i + 2], arr_y[i + 2], arr_z[i + 2]])

        first_dir_translated = -(start_pos - middle_pos)
        first_dir_translated -= (np.dot(first_dir_translated, middle_pos)) * middle_pos
        second_dir = end_pos - middle_pos
        second_dir -= (np.dot(second_dir, middle_pos)) * middle_pos

        angle = np.arccos(np.dot(first_dir_translated, second_dir) / (np.linalg.norm(first_dir_translated) * np.linalg.norm(second_dir)))
        sign = np.sign(np.dot(middle_pos, np.cross(first_dir_translated, second_dir)))

        deviation_angles.append(sign * angle)

    return np.array(deviation_angles)
