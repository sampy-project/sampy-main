from ...graph.spatial_2d import SpatialComponentsTwoDimensionalOrientedHexagons
from ...graph.topology import OrientedHexagonalGridOnSquare
from ...graph.vertex_attributes import BaseVertexAttributes
from ...utils.decorators import sampy_class

import numpy as np


@sampy_class
class HexGrid(BaseVertexAttributes,
              OrientedHexagonalGridOnSquare,
              SpatialComponentsTwoDimensionalOrientedHexagons):
    """
    This class is a Graph object corresponding to a lattice of hexagons.
    However, it is modified compared to the native SamPy one in order to
    make the interface between SamPy objects and QGIS easier. Namely, it
    allows to create geojson files that QGIS can read, allowing the user
    to visiualize and modify the lattice (for instance, by giving new 
    attributes to the hexagonal cells).

    todo
    """
    def __init__(self, **kwargs):
        pass

    @classmethod
    def azimuthal_from_corners(cls, bottom_left, bottom_right, top_left, top_right, 
                               radius=6_371.008, input_in_radians=False):
        # se begin with converting lat-lon to cartesian coordinates
        if input_in_radians:
            bottom_left_xyz = np.array([radius * np.cos(bottom_left[0]) * np.cos(bottom_left[1]),
                                        radius * np.cos(bottom_left[0]) * np.sin(bottom_left[1]),
                                        radius * np.sin(bottom_left[0])])
            bottom_right_xyz = np.array([radius * np.cos(bottom_right[0]) * np.cos(bottom_right[1]),
                                         radius * np.cos(bottom_right[0]) * np.sin(bottom_right[1]),
                                         radius * np.sin(bottom_right[0])])
            top_left_xyz = np.array([radius * np.cos(top_left[0]) * np.cos(top_left[1]),
                                     radius * np.cos(top_left[0]) * np.sin(top_left[1]),
                                     radius * np.sin(top_left[0])])
            top_right_xyz = np.array([radius * np.cos(top_right[0]) * np.cos(top_right[1]),
                                      radius * np.cos(top_right[0]) * np.sin(top_right[1]),
                                      radius * np.sin(top_right[0])])
        else:
            bottom_left_xyz = np.array([radius * np.cos(np.radians(bottom_left[0])) * np.cos(np.radians(bottom_left[1])),
                                        radius * np.cos(np.radians(bottom_left[0])) * np.sin(np.radians(bottom_left[1])),
                                        radius * np.sin(np.radians(bottom_left[0]))])
            bottom_right_xyz = np.array([radius * np.cos(np.radians(bottom_right[0])) * np.cos(np.radians(bottom_right[1])),
                                        radius * np.cos(np.radians(bottom_right[0])) * np.sin(np.radians(bottom_right[1])),
                                        radius * np.sin(np.radians(bottom_right[0]))])
            top_left_xyz = np.array([radius * np.cos(np.radians(top_left[0])) * np.cos(np.radians(top_left[1])),
                                    radius * np.cos(np.radians(top_left[0])) * np.sin(np.radians(top_left[1])),
                                    radius * np.sin(np.radians(top_left[0]))])
            top_right_xyz = np.array([radius * np.cos(np.radians(top_right[0])) * np.cos(np.radians(top_right[1])),
                                    radius * np.cos(np.radians(top_right[0])) * np.sin(np.radians(top_right[1])),
                                    radius * np.sin(np.radians(top_right[0]))])

        # compute the barycenter of those four corners and project it onto
        # the sphere. This gives us the center of our projection.
        barycenter = (bottom_left_xyz + bottom_right_xyz + top_left_xyz + top_right_xyz)/4.
        center_projection = (radius/np.linalg.norm(barycenter)) * barycenter

        # we project each corner onto the plane tangent to the sphere at center_projection
        proj_bl = (radius**2 / (center_projection[0] * bottom_left_xyz[0] + 
                                center_projection[1] * bottom_left_xyz[1] +
                                center_projection[2] * bottom_left_xyz[2])) * bottom_left_xyz
        proj_br = (radius**2 / (center_projection[0] * bottom_right_xyz[0] + 
                                center_projection[1] * bottom_right_xyz[1] +
                                center_projection[2] * bottom_right_xyz[2])) * bottom_right_xyz
        proj_tl = (radius**2 / (center_projection[0] * top_left_xyz[0] + 
                                center_projection[1] * top_left_xyz[1] +
                                center_projection[2] * top_left_xyz[2])) * top_left_xyz
        proj_tr = (radius**2 / (center_projection[0] * top_right_xyz[0] + 
                                center_projection[1] * top_right_xyz[1] +
                                center_projection[2] * top_right_xyz[2])) * top_right_xyz
        
        # we now define the unit vector on the plane. To minimize deformation around the
        # center, we use the two bottom corner to define X-direction at the center :
        x_dir_plane_unit = (proj_br - proj_bl) / np.linalg.norm(proj_br - proj_bl)
        center_projection_unit = center_projection / np.linalg.norm(center_projection)
        theta = 1./radius
        x_dir_sphere = radius * (np.cos(theta) * center_projection_unit + 
                                 np.sin(theta) * x_dir_plane_unit)
        x_dir_plane = (radius**2 / (center_projection[0] * x_dir_sphere[0] + 
                                    center_projection[1] * x_dir_sphere[1] +
                                    center_projection[2] * x_dir_sphere[2])) * x_dir_sphere