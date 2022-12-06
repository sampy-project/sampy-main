import numpy as np


class BaseGraphIntersection:
    """
    Define the basic attributes.
        graph_1: graph object
        graph_2: graph object

    """
    def __init__(self, graph_1=None, graph_2=None, **kwargs):
        if graph_1 is None:
            raise ValueError("No value has been given for kwargs 'graph_1'.")
        if graph_2 is None:
            raise ValueError("No value has been given for kwargs 'graph_2'.")
        self.graph_1 = graph_1
        self.graph_2 = graph_2

        # self.
        self.connections_g1_to_g2 = None
        self.weights_g1_to_g2 = None

        self.connections_g2_to_g1 = None
        self.weights_g2_to_g1 = None



