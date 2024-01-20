from ...graph.spatial_2d import SpatialComponentsTwoDimensionalOrientedHexagons
from ...graph.topology import OrientedHexagonalGridOnSquare
from ...graph.vertex_attributes import BaseVertexAttributes
from ...utils.decorators import sampy_class
from .jit_compiled_functions import keep_subgraph_from_array_of_bool_equi_weight

import numpy as np
import geojson


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
    
    def create_geojson(self, path_geojson):
        """
        Creates a geojson file containing all the 
        """
        list_features = []
        for index_hex in range(self.cell_vertices_lat_lon.shape[0]):
            list_features.append(geojson.Feature(geometry=geojson.Polygon([[(self.cell_vertices_lat_lon[index_hex, i%6, 1],
                                                                             self.cell_vertices_lat_lon[index_hex, i%6, 0]) for i in range(7)]]),
                                                                             properties={'id': index_hex}))
        final_geojson = geojson.FeatureCollection(list_features)
        with open(path_geojson, 'w') as f_out:
            f_out.write(geojson.dumps(final_geojson))
        print("geojson created")
        
    def save_graph(self, path_folder):
        """
        Save the graph in a folder. The new folder is created by the method. If something already
        exists at the given path, an exception is raised. Data are stored using .npy files, and the 
        resulting folder may be large depending on the size of the graph.

        :param path_folder: string or path object. Path to the folder to be created.  
        """
        pass

    def load_graph(self, path_folder):
        """
        TODO
        """
        pass

    def modify_from_csv(self, path_to_csv, name_col_with_id, sep=',', to_keep=None, 
                        dict_attribute_to_type=None):
        """
        TODO
        """
        # prepar dict in which data will be stored
        dict_attr_to_val = {name_col_with_id: []}
        if to_keep is not None:
            dict_attr_to_val[to_keep] = []
        if dict_attribute_to_type is not None:
            for key in dict_attribute_to_type.keys():
                dict_attr_to_val[key] = []
        
        with open(path_to_csv, 'r') as csv_in:
            for i, line in enumerate(csv_in):
                data = line.replace('\n', '').split(sep)

                if i == 0:
                    # get position in CSV of each column
                    dict_name_position = {name:pos for pos, name in enumerate(data)}
                    continue

                for col_name in dict_attr_to_val:
                    if col_name == name_col_with_id:
                        dict_attr_to_val[col_name].append(int(data[dict_name_position[col_name]]))
                    elif col_name == to_keep:
                        bool_val = data[dict_name_position[col_name]]
                        if (bool_val.lower() == 'true') or (bool_val == '1'):
                            bool_val = True
                        elif (bool_val.lower() == 'false') or (bool_val == '0'):
                            bool_val = False
                        else:
                            raise ValueError("Unexpected entry for boolean column:" + bool_val + '.')
                        dict_attr_to_val[to_keep].append(bool_val)
                    else:
                        if dict_attribute_to_type[col_name] == bool:
                            bool_val = data[dict_name_position[col_name]]
                            if (bool_val.lower() == 'true') or (bool_val == '1'):
                                bool_val = True
                            elif (bool_val.lower() == 'false') or (bool_val == '0'):
                                bool_val = False
                            else:
                                raise ValueError("Unexpected entry for boolean column:" + bool_val + '.')
                            dict_attr_to_val[col_name].append(bool_val)
                        else:
                            value = dict_attribute_to_type[col_name](data[dict_name_position[col_name]])
                            dict_attr_to_val[col_name].append(value)
                
        # the order of the vertices may be messed up, so we create a permutation that will sort 
        # everything as needed. Solution found on stack overflow for computing the inverse.
                            
        perm = np.array(dict_attr_to_val[name_col_with_id])
        inv = np.empty_like(perm)
        inv[perm] = np.arange(len(inv), dtype=inv.dtype)

        for key, val in dict_attr_to_val.items():
            if (key != name_col_with_id) and (key != to_keep):
                self.df_attributes[key] = np.array(val)[inv]

        new_connections, new_weights = keep_subgraph_from_array_of_bool_equi_weight(np.array(dict_attr_to_val[to_keep])[inv], 
                                                                                    self.connections)
        
        self.connections = new_connections
        self.weights = new_weights
        self.df_attributes = self.df_attributes[np.array(dict_attr_to_val[to_keep])[inv]]

        # create the new dict from id to index
        self.dict_cell_id_to_ind = dict()
        counter = 0
        for i, val in enumerate(np.array(dict_attr_to_val[to_keep])[inv]):
            if val:
                self.dict_cell_id_to_ind[i] = counter
                counter += 1
