from ...graph.topology import (BaseTopology,
                               OrientedHexagonalGridOnSquare)
from ...graph.vertex_attributes import BaseVertexAttributes, PeriodicAttributes
from ...graph.spatial_2d import SpatialComponentsTwoDimensionalOrientedHexagons
from ...utils.decorators import sampy_class
import xml.etree.ElementTree as ET
import numpy as np


@sampy_class
class GraphFromORMxml(BaseTopology,
                      BaseVertexAttributes,
                      PeriodicAttributes):
    """
    Class developed for the need of the Leighton Lab. The graph structure is read from an XML, as generated by Badr
    QGIS module.

    The obtained graph has a hexagonal structure (i.e. each vertex represents a hexagonal cell), where each vertex has
    an orientation. That is, a vertex has a north neighbour, a north-east one, a south-east one, etc... Vertices come
    with several attributes : 'K' measuring the carrying capacity, 'in-res' measuring how hard it is to get into a cell,
    'out-res' measuring how hard it is to leave a cell.

    Those two last resistance values are compressed into a 2D array of same shape as connections called
    prob_successful_move, that gives the probability for a movement along an edge to be successful.

    Finally, each cell belongs to some category, called a "super-cell". You can access them via methods below.
    """
    def __init__(self, path_to_xml=None):
        if path_to_xml is None:
            raise ValueError("A path to an ORM XML should be provided.")

        tree = ET.parse(path_to_xml)
        root = tree.getroot()

        # get super_cells info
        self.dict_super_cell = {}
        counter = 0
        for super_cell in root.findall('SuperCells'):
            temp_dict = {}
            for info in super_cell:
                temp_dict[info.tag] = info.text
            self.dict_super_cell[temp_dict['ID']] = {'index': counter,
                                                     'in_res': float(temp_dict['InResistance'])/100.,
                                                     'out_res': float(temp_dict['OutResistance'])/100.}
            counter += 1

        # read the various parameters to create a sampy graph
        counter = 0
        for cell in root.findall('AllCellData'):
            id_cell = cell.find('HEXID').text
            if id_cell not in self.dict_cell_id_to_ind:
                self.dict_cell_id_to_ind[id_cell] = counter
                counter += 1
            else:
                raise ValueError("The cell " + id_cell + " is defined two times in the xml.")

        self.connections = np.full((counter, 6), -1, dtype=np.int32)
        self.weights = np.full((counter, 6), -1.)
        self.prob_successful_move = np.full((counter, 6), -1.)
        self.df_attributes['K'] = np.full((counter,), 0.)
        self.df_attributes['in_res'] = np.full((counter,), 0.)
        self.df_attributes['out_res'] = np.full((counter,), 0.)
        self.df_attributes['super_cell'] = np.full((counter,), 0)
        self.df_attributes['easting'] = np.full((counter,), 0.)
        self.df_attributes['northing'] = np.full((counter,), 0.)
        array_nb_neighbours = np.full((counter,), 0, dtype=np.int8)

        for cell in root.findall('AllCellData'):
            index_cell = self.dict_cell_id_to_ind[cell.find('HEXID').text]
            for info in cell:

                if info.tag == 'N':
                    if info.text != 'b' and info.text in self.dict_cell_id_to_ind:
                        self.connections[index_cell][0] = self.dict_cell_id_to_ind[info.text]
                        array_nb_neighbours[index_cell] += 1
                if info.tag == 'NE':
                    if info.text != 'b' and info.text in self.dict_cell_id_to_ind:
                        self.connections[index_cell][1] = self.dict_cell_id_to_ind[info.text]
                        array_nb_neighbours[index_cell] += 1
                if info.tag == 'SE':
                    if info.text != 'b' and info.text in self.dict_cell_id_to_ind:
                        self.connections[index_cell][2] = self.dict_cell_id_to_ind[info.text]
                        array_nb_neighbours[index_cell] += 1
                if info.tag == 'S':
                    if info.text != 'b' and info.text in self.dict_cell_id_to_ind:
                        self.connections[index_cell][3] = self.dict_cell_id_to_ind[info.text]
                        array_nb_neighbours[index_cell] += 1
                if info.tag == 'SW':
                    if info.text != 'b' and info.text in self.dict_cell_id_to_ind:
                        self.connections[index_cell][4] = self.dict_cell_id_to_ind[info.text]
                        array_nb_neighbours[index_cell] += 1
                if info.tag == 'NW':
                    if info.text != 'b' and info.text in self.dict_cell_id_to_ind:
                        self.connections[index_cell][5] = self.dict_cell_id_to_ind[info.text]
                        array_nb_neighbours[index_cell] += 1

                elif info.tag == 'K':
                    self.df_attributes['K'][index_cell] = float(info.text)
                elif info.tag == 'supercell':
                    self.df_attributes['super_cell'][index_cell] = self.dict_super_cell[info.text]['index']
                    self.df_attributes['in_res'][index_cell] = self.dict_super_cell[info.text]['in_res']
                    self.df_attributes['out_res'][index_cell] = self.dict_super_cell[info.text]['out_res']
                elif info.tag == 'easting':
                    self.df_attributes['easting'][index_cell] = float(info.text)
                elif info.tag == 'northing':
                    self.df_attributes['northing'][index_cell] = float(info.text)

        # we now populate the weights array
        for i in range(self.connections.shape[0]):
            nb_neighbours = array_nb_neighbours[i]
            seen_neighb = 0
            if nb_neighbours == 0:
                continue
            for j in range(6):
                if self.connections[i][j] != -1:

                    self.prob_successful_move[i][j] = (1. - self.df_attributes['out_res'][i]) * \
                                                      (1. - self.df_attributes['in_res'][self.connections[i][j]])

                    seen_neighb += 1
                    if seen_neighb == nb_neighbours:
                        self.weights[i][j] = 1.
                        break
                    else:
                        self.weights[i][j] = float(seen_neighb)/float(nb_neighbours)

    def get_super_cell(self, name_super_cell):
        """
        Get a 1D array of bool arr_cell, where arr_cell[i] is True if and only if the cell of index i is a super-cell
        of the requested category.

        :param name_super_cell: String, the kind of super-cell you want the list of.

        :return: 1D array of bool
        """
        return self.df_attributes['super_cell'] == self.dict_super_cell[name_super_cell]['index']

    @property
    def list_super_cell_names(self):
        """
        Gives you the list of all the supercells present in your landscape.

        :return: a list of string
        """
        return [super_cell_name for super_cell_name in self.dict_super_cell]


class ORMLikeResistanceToMovement:
    """
    todo
    """
    def __init__(self, **kwargs):
        if not hasattr(self, 'connections'):
            raise AttributeError("The graph does not have a connections array.")
        if self.connections.shape[1] != 6:
            raise ValueError("The graph is not an hexagonal based grid.")

    def define_movement_resistance(self):
        """
        Set in_res and out_res to 0. for all vertices.
        """
        self.df_attributes['in_res'] = 0.
        self.df_attributes['out_res'] = 0.
        self.prob_successful_move = np.full(self.connections.shape, 1., dtype=float)

    def advanced_define_movement_resistance(self, in_res="in_res", out_res="out_res"):
        """
        Not implemented yet.
        """
        raise NotImplementedError("This method is not yet defined.")


@sampy_class
class ORMLikeHexGrid(OrientedHexagonalGridOnSquare,
                     BaseVertexAttributes,
                     SpatialComponentsTwoDimensionalOrientedHexagons,
                     ORMLikeResistanceToMovement):
    """
    todo
    """
    def __init__(self, nb_hex_x_axis=None, nb_hex_y_axis=None, **kwargs):
        pass