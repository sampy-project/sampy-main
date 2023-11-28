import numba as nb
import numpy as np


@nb.njit
def extract_deviation_angles(arr_x, arr_y, arr_z):
    """
    This function extracts the list of deviation angles from 3 lists
    of 3D coordinates. The points are assumed to lie on an unit sphere.
    """
    angles = np.full((arr_x.shape[0] - 2,), 0., dtype=float)
    for i in range(angles.shape[0]):
        # we define a few useful variables. Allocating all of these may
        # have a performance cost, but it improves readibility.
        start_pos =  np.array(arr_x[i],     arr_y[i],     arr_z[i])
        middle_pos = np.array(arr_x[i + 1], arr_y[i + 1], arr_z[i + 1])
        end_pos =    np.array(arr_x[i + 2], arr_y[i + 2], arr_z[i + 2])

        cross_start_mid = np.cross(start_pos, middle_pos)
        normal_vect = cross_start_mid / np.linalg.norm(cross_start_mid)

        angle_start_to_middle = np.arctan2(np.dot(cross_start_mid, normal_vect), np.dot(start_pos, middle_pos))
        # we compute the initial direction, that we then translate
        # to the middle point with parallel transport.
        # we first approximate the direction, then translate it, and
        # finally normalize it to be in the correct tangent plane.
        dir_from_start_pos = middle_pos - start_pos

        dir_from_middle_pos = end_pos - middle_pos