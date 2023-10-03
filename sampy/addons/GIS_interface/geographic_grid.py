from ...graph.topology import OrientedHexagonalGridOnSquare
from ...graph.vertex_attributes import BaseVertexAttributes
from ...graph.spatial_2d import SpatialComponentsTwoDimensionalOrientedHexagons
from ...utils.decorators import sampy_class


@sampy_class
class HexGrid(BaseVertexAttributes,
              OrientedHexagonalGridOnSquare,
              SpatialComponentsTwoDimensionalOrientedHexagons):
    """
    This class is a Graph object corresponding to a lattice of hexagons.
    However, it is modified compared to the native SamPy one in order to
    make the interface between SamPy objects and QGIS easier. Namely, it
    allows to create geojson files that QGIS can read, allowing the user
    to introduce visiualize and modify the lattice (for instance, by 
    giving new attributes to the hexagonal cells).

    todo
    """
    def __init__(self, **kwargs):
        pass

    