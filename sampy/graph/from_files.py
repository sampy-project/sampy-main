from ..pandas_xs.pandas_xs import DataFrameXS
from .misc import convert_graph_structure_to_dictionary

import numpy as np
import os
import shutil
import json
import glob


class SaveAndLoadSquareGrids:
    """
    Add to the square grids graphs the possibility to save them using their own special format.

    We also provide a Json
    """
    def __init__(self, **kwargs):
        pass

    def save(self, path_to_folder, erase_folder=True):
        """
        Save the graph structure in a folder using .npy files. The end result is not human readable.

        :param path_to_folder: Path to the folder. If it does not exists, the folder will be created.
        :param erase_folder: optional, boolean, default True. If True, any folder already existing at 'path_to_folder'
                             will be deleted.
        """
        if os.path.exists(path_to_folder):
            if not erase_folder:
                raise OSError("Something already exists at " + path_to_folder + '.')
            if not os.path.isdir(path_to_folder):
                raise OSError("The object at " + path_to_folder + " is not a directory. In doubt, we prefer not to " +
                              "delete it.")
            shutil.rmtree(path_to_folder)
        os.mkdir(path_to_folder)

        np.save(path_to_folder + '/connections.npy', self.connections)
        np.save(path_to_folder + '/weights.npy', self.weights)

        attributes_that_are_none = []
        for name in self.df_attributes.list_col_name:
            if self.df_attributes[name] is not None:
                np.save(path_to_folder + '/attr_' + name + '.npy', self.df_attributes[name])
            else:
                attributes_that_are_none.append(name)

        metadata_json = {'shape_0': self.shape[0],
                         'shape_1': self.shape[1],
                         'type': self.type,
                         'attributes_none': attributes_that_are_none}
        metadata = open(path_to_folder + '/metadata_json.json', 'w')
        metadata.write(json.dumps(metadata_json))
        metadata.close()

    @classmethod
    def load(cls, path_to_folder):
        """

        :param path_to_folder:
        :return:
        """
        if os.path.exists(path_to_folder):
            if not os.path.isdir(path_to_folder):
                raise OSError("The object at " + path_to_folder + " is not a directory.")
        else:
            raise OSError("Nothing at " + path_to_folder + '.')

        metadata = json.load(open(path_to_folder + '/metadata_json.json'))
        shape = (metadata['shape_0'], metadata['shape_1'])
        graph_type = metadata['type']

        graph = cls(shape=shape)
        graph.type = graph_type
        graph.shape = shape

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

    def save_as_json(self, path_to_json, erase_existing_file=False):
        """
        Save a SquareGrid graph as a JSON. This is not the recommended way of saving a

        :param path_to_json:
        :param erase_existing_file:
        """
        if os.path.exists(path_to_json):
            if not erase_existing_file:
                raise OSError("Something already exists at " + path_to_json + '.')
            os.remove(path_to_json)

        dict_graph_structure = convert_graph_structure_to_dictionary(self, save_vertices_index=True,
                                                                     add_to_metadata={'is_SquareGrid': True,
                                                                                      'shape_0': self.shape[0],
                                                                                      'shape_1': self.shape[1]})

        graph_json = open(path_to_json, 'w')
        graph_json.write(json.dumps(dict_graph_structure))
        graph_json.close()

    @classmethod
    def from_json(cls, path_to_json):
        """
        Load a SquareGrid graph from a JSON. Metadata section of the json should contain a flag 'is_SquareGrid" set to
        True and vertices index should be provided.

        :param path_to_json: string, path to a Json
        :return: Square Grid graph object.
        """
        graph_as_dict = json.load(open(path_to_json, 'r'))
        if 'is_SquareGrid' not in graph_as_dict['metadata'] or not graph_as_dict['metadata']['is_SquareGrid']:
            raise TypeError("The file in " + path_to_json + " is not a SquareGrid Json.")
        if not graph_as_dict['metadata']['vertices_index_provided']:
            raise TypeError("The file in " + path_to_json + " is supposedly a SquareGrid Json, yet vertices index " +
                            "are not provided.")
        shape = (graph_as_dict['metadata']['shape_0'], graph_as_dict['metadata']['shape_1'])
        graph = cls(shape=shape)
        graph.dict_cell_id_to_ind = {}
        for i in range(shape[0]):
            for j in range(shape[1]):
                graph.dict_cell_id_to_ind[(i, j)] = graph_as_dict['vertices'][str((i, j))]['index']

        for name in graph_as_dict['metadata']['empty_attributes']:
            graph.df_attributes[name] = None

        tmp_dict_attribute = {}
        for name in graph_as_dict['metadata']['non_empty_attributes']:
            tmp_dict_attribute[name] = [None for _ in range(graph_as_dict['metadata']['nb_vertices'])]

        for i in range(shape[0]):
            for j in range(shape[1]):
                index_vertex = graph.dict_cell_id_to_ind[(i, j)]
                for name in tmp_dict_attribute:
                    tmp_dict_attribute[name][index_vertex] = graph_as_dict['vertices'][str((i, j))][name]
                for k in range(graph_as_dict['metadata']['max_degree_vertex']):
                    if ('n' + str(k)) in graph_as_dict['vertices'][str((i, j))]:
                        graph.connections[index_vertex][k] = graph_as_dict['vertices'][graph_as_dict['vertices'][str((i, j))][('n' + str(k))]]['index']
                        graph.weights[index_vertex][k] = graph_as_dict['vertices'][str((i, j))][('w' + str(k))]
                    else:
                        graph.connections[index_vertex][k] = -1
                        graph.weights[index_vertex][k] = -1.

        for name in tmp_dict_attribute:
            graph.df_attributes[name] = tmp_dict_attribute[name]

        return graph


class FromJson:
    def __init__(self, **kwargs):
        pass
