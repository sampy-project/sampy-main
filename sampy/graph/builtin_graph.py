from .topology import (SquareGridWithDiagTopology,
                       SquareGridTopology,
                       SquareGridsConvertBetween1DArrayAnd2DArrays,
                       IcosphereTopology)
from .vertex_attributes import PeriodicAttributes, BaseVertexAttributes, AttributesFrom2DArraysSquareGrids
from .from_files import SaveAndLoadSquareGrids
from ..utils.decorators import sampy_class
from .misc import save_as_repository_include_metadata
from .jit_compiled_functions import keep_subgraph_from_array_of_bool_equi_weight

import os
import numpy as np
import glob
import json


@sampy_class
class SquareGridWithDiag(SquareGridWithDiagTopology,
                         BaseVertexAttributes,
                         PeriodicAttributes,
                         AttributesFrom2DArraysSquareGrids,
                         SaveAndLoadSquareGrids,
                         SquareGridsConvertBetween1DArrayAnd2DArrays):
    """
    Landscape graph. Grid of squares, diagonals included.
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class SquareGrid(SquareGridTopology,
                 BaseVertexAttributes,
                 PeriodicAttributes,
                 AttributesFrom2DArraysSquareGrids,
                 SaveAndLoadSquareGrids,
                 SquareGridsConvertBetween1DArrayAnd2DArrays):
    """
    Landscape graph. Grid of squares, diagonals excluded.
    """
    def __init__(self, **kwargs):
        pass


@sampy_class
class IcosphereGraph(BaseVertexAttributes,
                     IcosphereTopology):
    """
    Graph of choice for the study of species whose species distribution is big enough so that the shape of the earth
    has to be considered.
    """
    def __init__(self, **kwargs):
        pass

    def save(self, path_to_folder, erase_folder=True):
        """
        Save the graph structure in a folder using .npy files. The end result is not human-readable.

        :param path_to_folder: Path to the folder. If it does not exist, the folder will be created.
        :param erase_folder: optional, boolean, default True. If True, any folder already existing at 'path_to_folder'
                             will be deleted.
        """

        metadata_json = {'nb_sub': self.nb_sub,
                         'type': 'icosphere',
                         'radius': self.radius}

        save_as_repository_include_metadata(path_to_folder, metadata_json, self.df_attributes,
                                            self.connections, self.weights, erase_folder=erase_folder)

    @classmethod
    def load(cls, path_to_folder, strict_check=True):
        """
        Load the graph structure using a folder saved using the save method.

        :param path_to_folder: path to a folder where a graph icosphere is saved.
        :param strict_check: optional, boolean, default True. If true, check that the loaded graph as type 'icosphere'.
        :return: An instanciated IcosphereGraph object
        """
        if os.path.exists(path_to_folder):
            if not os.path.isdir(path_to_folder):
                raise OSError("The object at " + path_to_folder + " is not a directory.")
        else:
            raise OSError("Nothing at " + path_to_folder + '.')

        metadata = json.load(open(path_to_folder + '/metadata_json.json'))
        if metadata['type'] != 'icosphere' and strict_check:
            raise ValueError("According to the metadata, the graph is not of type icosphere.")
        nb_sub = int(metadata['nb_sub'])
        radius = float(metadata['radius'])

        graph = cls(nb_sub=3, radius=1.)
        graph.radius = radius
        graph.nb_sub = nb_sub

        graph.connections = np.load(path_to_folder + '/connections.npy')
        graph.weights = np.load(path_to_folder + '/weights.npy')

        for path in glob.glob(path_to_folder + '/*'):
            if os.path.basename(path).startswith('attr_'):
                name = os.path.basename(path).split('.')[0]
                name = name[5:]
                graph.df_attributes[name] = np.load(path)

        for name in metadata['attributes_none']:
            graph.df_attributes[name] = None

        return graph

    def keep_subgraph(self, array_vertices_to_keep):
        """
        Keep the specified vertices and keep the rest. Both attributes, connections and weights are updated accordingly.

        :param array_vertices_to_keep: 1d array of bool. array[i] is true if the vertex of index i should be kept.
        """
        new_connections, new_weights = keep_subgraph_from_array_of_bool_equi_weight(array_vertices_to_keep,
                                                                                    self.connections)
        self.connections = new_connections
        self.weights = new_weights
        self.df_attributes = self.df_attributes[array_vertices_to_keep]
