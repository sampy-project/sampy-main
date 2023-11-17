from .base_graph_intersection import BaseGraphIntersection
from .graph_intersection_geometry import GraphIntersectionConvexPolygons
from ..utils.decorators import sampy_class


@sampy_class
class Intersection2DGraphsConvexPolygonalCells(BaseGraphIntersection,
                                              GraphIntersectionConvexPolygons):
    """
    Quick guide on how to use this class.
    -----------------------------------------------------------------------------------
    The quickest way to use this class is to start with 2 builtin graphs g1 and g2 
    taken among the list: 

        [SquareGrid, SquareGridWithDiag, OrientedHexagonalLattice]. 

    Any combination will work. Then, create for each of those graphs their "geometric 
    features" using first the method 'set_coords_from_vector' (which associate 2D 
    coordinates to each vertex), and then the method 'create_hexagon_vertices' or 
    'create_square_vertices' (which associate polygonal cells to each vertex of the 
    graphs).
    Once this is done, you will have two graphs with their geometric features, and
    you can now use this Intersection class as follows:

    
    intersection = Intersection2DGraphsConvexPolygonalCells(graph_1=g1,
                                                            graph_2=g2)
    intersection.compute_intersection_same_nb_vertices()


    With this, the intersection object is fully set up, and you can convert 1D arrays
    from 1 graph to 1D arrays of the other using the convertion methods. 


    More detailed documentation
    -----------------------------------------------------------------------------------
    This class requires a certain number of steps to be performed before it
    becomes usable.

        1) This class constructor takes a pair of graphs, and each one of them should 
           have 'geometric features'. That is, each vertex of each graph should have
           2D coordinates stored in their attributes dataframe, and each vertex should
           have an associated convex polygon (called cell) whose centroid is the coords
           of the vertex.
        2) Those cells are assumed to have the same number of edges within each graph, 
           and to be stored in a 3D arrays of shape 
                (nb_vert_graph_j, nb_edges_poly_graph_j, 2).
           If those 

           TO BE FINISHED
    """
    def __init__(self, **kwargs):
        pass

