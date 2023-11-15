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
                               radius=6_371.008, input_in_radians=False, shape=None,
                               cell_area=None):
        """
        Create a sampy-graph object, whose underlying structure is a Hexagonal Grid.
        The coordinates of the hexagons centroids and vertices are given in Lat-Lon.
        Those geographic coordinates are computed using a projection 

        todo
        """
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
        x_dir_plane = x_dir_plane - center_projection
        y_dir_plane = np.cross(center_projection_unit, x_dir_plane)

        # we compute the coordinates of the 4 corners in the plane spanned by x_dir_plane 
        # and y_dir_plane
        proj_bl_plane = np.array([np.dot(x_dir_plane, proj_bl), np.dot(y_dir_plane, proj_bl)])
        proj_br_plane = np.array([np.dot(x_dir_plane, proj_br), np.dot(y_dir_plane, proj_br)])
        proj_tl_plane = np.array([np.dot(x_dir_plane, proj_tl), np.dot(y_dir_plane, proj_tl)])
        proj_tr_plane = np.array([np.dot(x_dir_plane, proj_tr), np.dot(y_dir_plane, proj_tr)])

        # in order to create the hexagonal grid, we now need to define the boundaries of a 
        # rectangle containing the four corners on the plane
        x_min_rectangle = min(proj_bl_plane[0], proj_br_plane[0], proj_tl_plane[0], proj_tr_plane[0])
        x_max_rectangle = max(proj_bl_plane[0], proj_br_plane[0], proj_tl_plane[0], proj_tr_plane[0])
        y_min_rectangle = min(proj_bl_plane[1], proj_br_plane[1], proj_tl_plane[1], proj_tr_plane[1])
        y_max_rectangle = max(proj_bl_plane[1], proj_br_plane[1], proj_tl_plane[1], proj_tr_plane[1])

        # we now need to define the shape of the hexagonal grid (i.e., the number of hexagonal cells
        # per side of the rectangle). This is either given by the user using the kwarg shape, or 
        # computed here if the user provided the area of one cell.
        if shape is None:
            if cell_area is None:
                raise ValueError("No parameters has been given to compute the shape of the grid." +
                                 " Either use kwarg 'shape' to give it directly, or use the kwarg" +
                                 " 'cell_area' to have it compute by this method.")
            try:
                dist_between_centroids = np.sqrt((2/np.sqrt(3)) * float(cell_area))
            except ValueError:
                raise ValueError("The values given for the area of a cell cannot be casted to a float.")
            
            shape = (int(np.ceil((x_max_rectangle - x_min_rectangle) / ((np.sqrt(3)/2) * dist_between_centroids))),
                     int(np.ceil((y_max_rectangle - y_min_rectangle) / dist_between_centroids)))
            
        else:
            dist_between_centroids = (y_max_rectangle - y_min_rectangle) / float(shape[1])
        
        # we now create the hexagonal grid
        hex_grid = cls(nb_hex_x_axis=shape[0], nb_hex_y_axis=shape[1])
        hex_grid.set_coords_from_vector((x_min_rectangle, y_min_rectangle), (0., dist_between_centroids))
        hex_grid.create_hexagon_vertices()

        # we create the 3D coordinates of the centroids
        hex_centroids_x = hex_grid.df_attributes['coord_x'] * x_dir_plane[0] + \
                          hex_grid.df_attributes['coord_y'] * y_dir_plane[0] + \
                          center_projection[0]
        hex_centroids_y = hex_grid.df_attributes['coord_x'] * x_dir_plane[1] + \
                          hex_grid.df_attributes['coord_y'] * y_dir_plane[1] + \
                          center_projection[1]
        hex_centroids_z = hex_grid.df_attributes['coord_x'] * x_dir_plane[2] + \
                          hex_grid.df_attributes['coord_y'] * y_dir_plane[2] + \
                          center_projection[2]
        
        print(x_dir_plane)
        print(y_dir_plane)
        print(center_projection)
        
        # we normalize those coordinates to compute the lat-lon couples
        norm = np.sqrt(hex_centroids_x**2 + hex_centroids_y**2 + hex_centroids_z**2)
        hex_centroids_x_normalized = np.longdouble(hex_centroids_x / norm)
        hex_centroids_y_normalized = np.longdouble(hex_centroids_y / norm)
        hex_centroids_z_normalized = np.longdouble(hex_centroids_z / norm)

        # we now compute the lat-lon coordinates of the centroids
        hex_grid.df_attributes['lat_hex_centroids'] = (180*(np.pi/2 - np.arccos(hex_centroids_z_normalized))/np.pi).astype(np.float64)
        hex_grid.df_attributes['lon_hex_centroids'] = (180*np.arctan2(hex_centroids_y_normalized, hex_centroids_x_normalized)/np.pi).astype(np.float64)

        # we do the same thing for the vertices
        hex_vertices_x = hex_grid.cell_vertices[:, :, 0] * x_dir_plane[0] + \
                         hex_grid.cell_vertices[:, :, 1] * y_dir_plane[0] + \
                         center_projection[0]
        hex_vertices_y = hex_grid.cell_vertices[:, :, 0] * x_dir_plane[1] + \
                         hex_grid.cell_vertices[:, :, 1] * y_dir_plane[1] + \
                         center_projection[1]
        hex_vertices_z = hex_grid.cell_vertices[:, :, 0] * x_dir_plane[2] + \
                         hex_grid.cell_vertices[:, :, 1] * y_dir_plane[2] + \
                         center_projection[2]
        norm = np.sqrt(hex_vertices_x**2 + hex_vertices_y**2 + hex_vertices_z**2)
        hex_vertices_x_normalized = np.float64(hex_vertices_x / norm)
        hex_vertices_y_normalized = np.float64(hex_vertices_y / norm)
        hex_vertices_z_normalized = np.float64(hex_vertices_z / norm)
        hex_grid.cell_vertices_lat_lon = np.dstack([(180*(np.pi/2 - np.arccos(hex_vertices_z_normalized))/np.pi).astype(np.float64),
                                                    (180*np.arctan2(hex_vertices_y_normalized, hex_vertices_x_normalized)/np.pi).astype(np.float64)])
        
        return hex_grid