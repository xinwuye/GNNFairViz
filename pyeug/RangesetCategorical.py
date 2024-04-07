import pandas as pd
import numpy as np
import holoviews as hv

from shapely.ops import triangulate
from shapely.geometry import MultiPoint, MultiLineString, Polygon, MultiPolygon, LineString
from shapely.ops import unary_union


def _min_max_edge(polygon):
    '''Compute the length of the longest edge of a polygon.
    
    Parameters
    ----------
    polygon: shapely.Polygon
        The polygon to be tested.
        
    Returns
    -------
    max_edge: float
        Length of the longest edge.
    '''
    # return max([LineString([p,q]).length for p,q in zip(polygon.boundary.coords[:-1],polygon.boundary.coords[1:])])
    edge_lengths = [LineString([p,q]).length for p,q in zip(polygon.boundary.coords[:-1],polygon.boundary.coords[1:])]
    return min(edge_lengths), max(edge_lengths)


def pre_compute_contours(colormap, xy, layer, groups):
    points_2d = xy[layer]
    max_edge_len_lst = []
    min_edge_len_lst = []
    polygons_lst = []
    max_edges_lst = []
    for s, c in colormap.items():
        # sens is a np array, find the index of the values in sens that are equal to s
        idxs = np.where(groups == s)[0]
        points    = MultiPoint(points_2d[idxs])
        polygons = triangulate(points)

        min_max_edges = [_min_max_edge(polygon) for polygon in polygons]
        max_edges = [max_edge for min_edge, max_edge in min_max_edges]
        min_edges = [min_edge for min_edge, max_edge in min_max_edges]
        max_edge_len = max(max_edges)
        min_edge_len = min(min_edges)

        max_edge_len_lst.append(max_edge_len)
        min_edge_len_lst.append(min_edge_len)
        polygons_lst.append(polygons)
        max_edges_lst.append(max_edges)

    return min(min_edge_len_lst), max(max_edge_len_lst), polygons_lst, max_edges_lst


def compute_contours(colormap, polygons_lst, max_edges_lst, threshold):
    poly_pos = [[],[]]
    poly_color = []
    # for c in colormap:
    for i, (s, c) in enumerate(colormap.items()):
        polygons = polygons_lst[i]
        max_edges = max_edges_lst[i]
        poly = unary_union([polygon for j, polygon in enumerate(polygons) if max_edges[j] <= threshold]) # preserve
        
        # points are in one or multiple polygons
        if not poly.is_empty:
            # convert single polygons to multipolygon
            poly = MultiPolygon([poly]) if isinstance(poly, Polygon) else poly
            
            for p in poly.geoms:
                # ensure uniform boundary representation
                boundary = MultiLineString([p.boundary]) if isinstance(p.boundary, LineString) else p.boundary
                # print(boundary)
                # bb = boundary[0].coords.xy
                bb = boundary.geoms[0].coords.xy
                holes_x = [h.coords.xy[0].tolist() for h in list(boundary.geoms)[1:]]
                holes_y = [h.coords.xy[1].tolist() for h in list(boundary.geoms)[1:]]
                poly_pos[0].append([[bb[0].tolist()]+holes_x])
                poly_pos[1].append([[bb[1].tolist()]+holes_y]) 
                
                poly_color.append(c)
                
    return pd.DataFrame({'xs': poly_pos[0], 'ys': poly_pos[1], 'color': poly_color})


# def compute_contours(colormap, points_2d, sens_selected, threshold):
#     '''Compute the rangeset contours for a given variable.
    
#     Parameters
#     ----------
#     var: string
#         A variable contained in the DataFrame.
#     val_range: tuple of floats (min,max)
#         Custom (min,max)-range used as boundaries for the discretization.
#     bins: int
#         Number of bins for the discretization.
        
#     Returns
#     -------
#     polygons: pandas.DataFrame
#         DataFrame matching the multi_polygons data format of bokeh. Represents the boundary contours of the rangeset.
#     points: pandas.DataFrame
#         DataFrame matching the scatter data format of bokeh. Contains colored scatter data with respective scaling 
#         for inliers and outliers.
#     bounds: array
#         Boundaries used during discretization.
#     cnt_in: array of int
#         Number of inlying points per bin.
#     cnt_out: array of int
#         Number of outlying points per bin.
#     '''

#     poly_pos = [[],[]]
#     poly_color = []

#     # colors = hv.Cycle('Category20').values
#     # sens_selected_unique = np.unique(sens_selected)
#     # colormap = dict(zip(sens_selected_unique, colors[:len(sens_selected_unique)]))
#     # print('colormap: ', colormap)

#     # for c in colormap:
#     for s, c in colormap.items():
#         # sens is a np array, find the index of the values in sens that are equal to s
#         idxs = np.where(sens_selected == s)[0]
#         points    = MultiPoint(points_2d[idxs])
#         polygons = triangulate(points)
#         min_max_edges = [_min_max_edge(polygon) for polygon in polygons]
#         max_all_edges = max([max_edge for min_edge, max_edge in min_max_edges])
#         min_all_edges = min([min_edge for min_edge, max_edge in min_max_edges])
#         threshold = (max_all_edges - min_all_edges) / 2
#         poly      = unary_union([polygon for i, polygon in enumerate(polygons) if min_max_edges[i][1] <= threshold]) # preserve
#         # print('poly.is_empty: ', poly.is_empty)
        
#         # points are in one or multiple polygons
#         if not poly.is_empty:
#             # convert single polygons to multipolygon
#             poly = MultiPolygon([poly]) if isinstance(poly, Polygon) else poly
            
#             for p in poly.geoms:
#                 # ensure uniform boundary representation
#                 boundary = MultiLineString([p.boundary]) if isinstance(p.boundary, LineString) else p.boundary
#                 # print(boundary)
#                 # bb = boundary[0].coords.xy
#                 bb = boundary.geoms[0].coords.xy
#                 holes_x = [h.coords.xy[0].tolist() for h in list(boundary.geoms)[1:]]
#                 holes_y = [h.coords.xy[1].tolist() for h in list(boundary.geoms)[1:]]
#                 poly_pos[0].append([[bb[0].tolist()]+holes_x])
#                 poly_pos[1].append([[bb[1].tolist()]+holes_y]) 
                
#                 poly_color.append(c)
                
#     return pd.DataFrame({'xs': poly_pos[0], 'ys': poly_pos[1], 'color': poly_color}), min_all_edges, max_all_edges