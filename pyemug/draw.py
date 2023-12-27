from . import util
import numpy as np
import holoviews as hv
from holoviews import opts
# import os
# import torch
import numpy as np
# from sklearn.manifold import TSNE
# from holoviews.streams import Stream, param
# from holoviews import streams
# import datashader as ds
# import datashader.transfer_functions as tf
# import pandas as pd
# from datashader.bundling import connect_edges, hammer_bundle
from holoviews.operation.datashader import datashade, bundle_graph
# import panel as pn
from bokeh.models import CustomJSTickFormatter
# from tqdm import tqdm
from . import RangesetCategorical
from scipy.spatial.distance import cdist

hv.extension('bokeh')

PERPLEXITY = 15 # german
SEED = 42

def draw_graph_view(xy, source_id, target_id, layer, sens, 
                    sens_names, sens_name, bundle, sampled_alpha, colors, index):
    current_xy = xy[layer]
    x = current_xy[:,0]
    y = current_xy[:,1]
    node_indices = np.arange(0, len(x))
    # colors = hv.Cycle('Category20').values
    # colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', '#4e7670']
    # find the index of sens_name in sens_names
    sens_name_idx = sens_names.index(sens_name)
    sens_selected = sens[sens_name_idx]
    sens_selected_unique = np.unique(sens_selected)
    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
    # sampled_alpha = np.zeros(len(x))
    nodes = hv.Nodes((x, y, node_indices, sens_selected, sampled_alpha), vdims=['Sens', 'sampled_alpha']) \
        .opts(opts.Nodes(alpha='sampled_alpha', nonselection_alpha=0))
    graph = hv.Graph(((source_id, target_id), nodes, )) \
        # .opts(cmap=colors, node_size=4, edge_line_width=1,
        #         node_line_color='gray', node_color='Sens')
        # .opts(opts.Graph(edge_line_width=0.1, inspection_policy='nodes', tools=['hover', 'box_select'],
        #         edge_hover_line_color='green', node_hover_fill_color='red'))
    graph.nodes.opts(opts.Nodes(color='Sens', alpha='sampled_alpha', cmap=colormap, size=3, line_color=None,))
    _ = graph.opts(cmap=colormap, node_size=4, edge_line_width=0.1,
                node_line_color='gray', node_color='Sens', node_alpha='sampled_alpha')
    if bundle:
        g = bundle_graph(graph)
    else:
        g = graph

    # draw poly
    poly_df = RangesetCategorical.compute_contours(colormap, current_xy, sens_selected, threshold=10)
    # print('poly_df: ', poly_df)
    polys = hv.Polygons([{('x', 'y'): list(zip(poly_df.iloc[i]['xs'][0][0], poly_df.iloc[i]['ys'][0][0])), 'level': poly_df.iloc[i]['color']} for i in range(len(poly_df))], vdims='level')
    polys.opts(color='level', line_width=0, alpha=0.5)

    # # draw distribution
    # xdist = None
    # ydist = None
    # for ssu in sens_selected_unique:
    #     points = nodes.select(Sens=ssu)
    #     tmp_xdist = hv.Distribution(points, kdims=['x']).opts(color=colormap[ssu])
    #     tmp_ydist = hv.Distribution(points, kdims=['y']).opts(color=colormap[ssu])
    #     if xdist is None:
    #         xdist = tmp_xdist
    #         ydist = tmp_ydist
    #     else:
    #         xdist *= tmp_xdist
    #         ydist *= tmp_ydist

    ret = (g * polys * g.nodes).opts(
        opts.Nodes(color='Sens', alpha='sampled_alpha', size=4, height=300, width=300, cmap=colors, 
            # legend_position='right', 
            show_legend=False,
            tools=['lasso_select', 'box_select']),
        opts.Graph(edge_line_width=0.1, inspection_policy='nodes', 
            edge_hover_line_color='green', node_hover_fill_color='red')
        ).opts(xlabel='x_graph_view', ylabel='y_graph_view', border=0)
    # ret = polys
    # ret = ret << xdist.opts(width=125, xaxis=None, yaxis=None, border=0) << ydist.opts(width=125, xaxis=None, yaxis=None, border=0)

    return ret


def draw_distribution(xy, layer, sens, sens_names, sens_name, colors, x_or_y):
    sens_name_idx = sens_names.index(sens_name)
    sens_selected = sens[sens_name_idx]
    sens_selected_unique = np.unique(sens_selected)
    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
    current_xy = xy[layer]
    x = current_xy[:,0]
    y = current_xy[:,1]
    node_indices = np.arange(0, len(x))
    nodes = hv.Nodes((x, y, node_indices, sens_selected), vdims=['Sens'])
    # draw distribution
    dist = None
    for ssu in sens_selected_unique:
        points = nodes.select(Sens=ssu)
        tmp_dist = hv.Distribution(points, kdims=[x_or_y]).opts(color=colormap[ssu])
        if dist is None:
            dist = tmp_dist
            # ydist = tmp_ydist
        else:
            dist *= tmp_dist
            # ydist *= tmp_ydist

    if x_or_y == 'y':
        return dist.opts(width=60, xaxis=None, yaxis=None, border=0)
    else:
        return dist.opts(height=60, xaxis=None, yaxis=None, border=0)


def draw_legend(sens, sens_names, sens_name, colors):
    # colors = hv.Cycle('Category20').values
    # colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', '#4e7670']
    # use point and text elements to draw the legend, which should correspond to unique values in sens[sen_name_idx] and node colors in graph.nodes
    # get the unique values in sens[sen_name_idx]
    sens_name_idx = sens_names.index(sens_name)
    unique_sens = np.unique(sens[sens_name_idx])
    n_unique_sens = len(unique_sens)
    # draw the pointss vertically
    x_legend = np.zeros(n_unique_sens)
    y_legend = np.arange(0, n_unique_sens)
    # use linspace to make the points evenly spaced
    # y_legend = np.linspace(0, 0.1, len(unique_sens))
    # reverse the order of the y_legend
    y_legend = y_legend[::-1]
    # draw the texts
    texts = unique_sens
    # draw the points
    points = hv.Points((x_legend, y_legend, texts), vdims=['Sens']) \
        .opts(opts.Points(size=10, color='Sens', cmap=colors, nonselection_alpha=0.1, height=30 * n_unique_sens, 
            width=100, xaxis=None, yaxis=None, border=0, padding=0.5))
    # draw the texts
    for i, text in enumerate(texts):
        points = points * hv.Text(x_legend[i] + 0.6, y_legend[i], text)
    points.opts(show_legend=False, toolbar=None)

    return points


def draw_edge_legend():
    # colors_edge is a dict: {edge_type: color}
    colors_edge = {1: '#FE0000', 2: '#F4E0B9', 3: '#7D7463'}
    # draw the edge legend
    # use line and text elements to draw the legend
    # draw the lines
    # draw hv.Path for each line
    labels = ['unfair', 'fair', 'unfair & fair']
    paths = []
    text = None
    for i, edge_type in enumerate(colors_edge.keys()):
        paths.append([(0, i, edge_type), (0.5, i, edge_type)])
        if text is None:
            text = hv.Text(0.6, i, labels[i])
        else:
            text = text * hv.Text(0.6, i, labels[i])
    path = hv.Path(paths, vdims=['edge_type'])

    ret = (path * text).opts(
        opts.Path(color='edge_type', height=30 * len(colors_edge), width=100, cmap=colors_edge,
            xaxis=None, yaxis=None),
        opts.Text(text_font_size='10pt', text_align='left', text_baseline='middle', show_legend=False, toolbar=None)
        ).opts(
            border=0, padding=0.5, xlim=(-0.5, 3.5), ylim=(-1, len(colors_edge))
        )

    return ret


def draw_explanation_view(xy, layer, explain_induced_node_indices, 
    explain_induced_node_indices_unfair_pure, explain_induced_node_indices_fair_pure, explain_induced_node_indices_intersection,
    explain_induced_node_hops, source_explained, target_explained,
    edge_type, sens, sens_names, sens_name, layout, colors): 
    # get the min and max of x and y
    # x_min = x_induced.min()
    # x_max = x_induced.max()
    # y_min = y_induced.min()
    # y_max = y_induced.max()
    x_min = -1.
    x_max = 1.
    y_min = -1.
    y_max = 1.

    if layout == 'tsne':   
        current_xy = xy[layer]
        # draw the explaining induced subgraph
        x_induced = current_xy[explain_induced_node_indices.tolist(), 0]
        y_induced = current_xy[explain_induced_node_indices.tolist(), 1]
        # rescale x_induced and y_induced to [-x_min, x_max] and [-y_min, y_max]
        if x_induced.size > 0:
            x_induced_min = x_induced.min()
            x_induced_max = x_induced.max()
            y_induced_min = y_induced.min()
            y_induced_max = y_induced.max()
            x_induced = (x_induced - x_induced_min) / (x_induced_max - x_induced_min) * (x_max - x_min) + x_min
            y_induced = (y_induced - y_induced_min) / (y_induced_max - y_induced_min) * (y_max - y_min) + y_min
    elif layout == 'hop-wise':
        extra_layer = 2
        x_induced, y_induced, unique_hops_max = util.hopwise_force_directed_layout(explain_induced_node_indices, explain_induced_node_hops, 
            explain_induced_node_indices_unfair_pure, explain_induced_node_indices_fair_pure, explain_induced_node_indices_intersection,
            source_explained, target_explained, extra_layer=extra_layer)
        # print(x_induced)
        # draw bounding boxes for each hop, equally divide the x into unique_hops_max + 1 parts, leaving a little space between each hop
        rects_data = []
        if unique_hops_max >= 0:
            padding = ((x_max - x_min) / (unique_hops_max + 1) - (x_max - x_min) / (unique_hops_max + 1 + extra_layer)) / 4
            for hop in range(unique_hops_max + 1):
                x_left = x_min + (x_max - x_min) / (unique_hops_max + 1) * hop + padding
                x_right = x_min + (x_max - x_min) / (unique_hops_max + 1) * (hop + 1) - padding
                y_bottom = y_min * 1.05
                y_top = y_max * 1.05
                rects_data.append((x_left, y_bottom, x_right, y_top))
        rects = hv.Rectangles(rects_data).opts(color=None)
        if unique_hops_max >= 0:
            # create text elements for each rectangle in rects, put the text elements on the top center of each rectangle
            min_hop = explain_induced_node_hops.min()
            for i in range(len(rects_data)):
                x_left, y_bottom, x_right, y_top = rects_data[i]
                x_center = (x_left + x_right) / 2
                label = 'selected' if i + min_hop == 0 else 'hop {}'.format(int(i + min_hop))
                rect_texts = hv.Text(x_center, y_top * 1.1, label)
                rects = rects * rect_texts

    sens_name_idx = sens_names.index(sens_name)
    sens_selected = sens[sens_name_idx]
    sens_selected_unique = np.unique(sens_selected)
    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
    node_sens = sens_selected[explain_induced_node_indices.tolist()]
    # print(x_induced.shape)
    # print(explain_induced_node_indices.shape)
    nodes_induced = hv.Nodes((x_induced, y_induced, explain_induced_node_indices, node_sens), 
        vdims=['Sens']) 
    # nodes_induced = hv.Nodes((x_induced, y_induced, explain_induced_node_indices, explain_induced_node_hops), 
    #     vdims=['explain_induced_node_hops']) 
    # print(len(edge_type))
    # print(len(source_explained))
    # print(edge_type)
    graph_induced = hv.Graph(((source_explained, target_explained, edge_type), nodes_induced), vdims=['et'])

    # colors = hv.Cycle('Category20').values
    # colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', '#4e7670']
    # colors_edge = [colors[2], colors[5], colors[8]]
    colors_edge = {1: '#FE0000', 2: '#F4E0B9', 3: '#7D7463'} # 1: unfair, 2: fair, 3: intersection
    graph_induced.nodes.opts(opts.Nodes(color='Sens', cmap=colormap, size=3, line_color=None, show_legend=False))
    graph_induced.opts(cmap=colormap, node_size=4, edge_line_width=0.5,
                node_line_color='gray', node_color='Sens', show_legend=False)
    graph_induced.opts(opts.Graph(edge_color='et', 
        edge_cmap=colors_edge
        ))

    if layout == 'hop-wise':
        ret = (graph_induced * graph_induced.nodes * rects) \
            .opts(xaxis=None, yaxis=None, framewise=True, shared_axes=False, xlabel='x_explanation_view', ylabel='y_explanation_view',
                xlim=(x_min * 1.1, x_max * 1.1), ylim=(y_min * 1.1, y_max * 1.3),
                ).opts(opts.Graph(tools=['lasso_select', 'box_select'])) \
                .opts(opts.Nodes(show_legend=False))
    else:
        ret = (graph_induced * graph_induced.nodes) \
            .opts(xaxis=None, yaxis=None, framewise=True, shared_axes=False, xlabel='x_explanation_view', ylabel='y_explanation_view',
                xlim=(x_min * 1.1, x_max * 1.1), ylim=(y_min * 1.1, y_max * 1.3),
                ).opts(opts.Graph(tools=['lasso_select', 'box_select'])) \
                .opts(opts.Nodes(show_legend=False))

    return ret

def draw_metric_view(metrics, selected_metrics):
    curve_dict = {selected_metric: draw_metric_curve(metrics, selected_metric) 
        for selected_metric in selected_metrics}
    if curve_dict:
        ndoverlay = hv.NdOverlay(curve_dict)
    else:
        # ndoverlay = hv.Curve([])
        ndoverlay = hv.NdOverlay({'nothing': hv.Curve([], '#operation', 'metric')})
    return ndoverlay


def draw_metric_curve(metrics, selected_metric):
    # split the string selected_metric by '-'
    # get the first element
    layer, col = selected_metric.split('-', 1)
    metric = metrics[int(layer)]
    current_metric = metric[col]
    current_metric_formatted = list(zip(current_metric.index, current_metric.values))
    if len(current_metric_formatted) == 1:
        current_metric_formatted.append((0.05, current_metric_formatted[0][1]))
    curve = hv.Curve(current_metric_formatted, '#operation', 'metric')
    return curve


def draw_attr_hist(hop_indicator_induced_unfair, hop_indicator_induced_fair, feat, n_bins, i_col, max_hop, colors):
    # reshap feat to 2d array
    if len(feat.shape) > 2:
        feat_2d = feat.reshape(-1, feat.shape[-1])
    else:
        feat_2d = feat
    # colors = hv.Cycle('Category20').values
    # colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', '#4e7670']
    colors = colors[:: -1]
    rev_hop_indicator_induced_unfair = hop_indicator_induced_unfair[:: -1]
    rev_hop_indicator_induced_fair = hop_indicator_induced_fair[:: -1]

    ret = None
    col = feat_2d[: , i_col]
    frequencies, edges = np.histogram(col, n_bins)
    y_max = frequencies.max() * 1.1
    x_min = col.min()
    x_max = col.max()
    # max_hop = len(rev_hop_indicator_total) - 1

    # for i, hop in enumerate(rev_hop_indicator_total):
    for i in range(max_hop + 1):
        if rev_hop_indicator_induced_fair:
            hop = rev_hop_indicator_induced_fair[i]
            data = col[hop]
            f, e = np.histogram(data, bins=edges)
            hist = hv.Histogram((e, -f), label='selected' if i == max_hop else 'hop {}'.format(max_hop - i)) \
                .opts(fill_color=colors[i])
        else:
            hist = hv.Histogram(([], []), label='selected' if i == max_hop else 'hop {}'.format(max_hop - i))

        if ret is None:
            ret = hist
        else:
            ret = ret * hist

    # for i, hop in enumerate(rev_hop_indicator_induced):
    for i in range(max_hop + 1):
        if rev_hop_indicator_induced_unfair:
            hop = rev_hop_indicator_induced_unfair[i]
            data = col[hop]
            f, e = np.histogram(data, bins=edges)
            hist = hv.Histogram((e, f), ) \
                .opts(fill_color=colors[i])
        else:
            hist = hv.Histogram(([], []), )
        ret = ret * hist
        
    if ret:
        ret.opts(yformatter=CustomJSTickFormatter(code="""
            return Math.abs(tick)
        """), xlabel='Attr. {}'.format(str(int(i_col))), 
        ylabel='Freq.')
    else:
        ret = hv.Histogram(([], [])) * hv.Histogram(([], []))

    ret.opts(ylim=(-y_max, y_max), xlim=(x_min, x_max), shared_axes=False, legend_position='right', width=400, )

    return ret


def draw_feature_view(feat, hop_indicator_computation_graph, hop_indicator_induced_unfair, hop_indicator_induced_fair, max_hop):
    # reshap feat to 2d array
    if len(feat.shape) > 2:
        feat_2d = feat.reshape(-1, feat.shape[-1])
    else:
        feat_2d = feat
    # create a 4 by max_hop 0 np array, 0th row: unfair, 1st row: fair, 2nd row: computation graph, 3rd row: total
    dist_heatmap = np.zeros((4, max_hop))
    if hop_indicator_computation_graph:
        selected_feat = feat_2d[hop_indicator_computation_graph[0]]
        total_dist_mat = cdist(feat_2d, selected_feat, 'euclidean')
        dist_heatmap[3, : ] = total_dist_mat.mean()
        for i in range(1, max_hop + 1):
            dist_heatmap[0, i - 1] = total_dist_mat[hop_indicator_induced_unfair[i]].mean()
            dist_heatmap[1, i - 1] = total_dist_mat[hop_indicator_induced_fair[i]].mean()
            dist_heatmap[2, i - 1] = total_dist_mat[hop_indicator_computation_graph[i]].mean()

    row_names = ['unfair explanation', 'fair explanation', 'computation graph', 'total']

    data = [(str(j), row_names[i], dist_heatmap[i, j]) for i in range(4) for j in range(max_hop)]
    ret = hv.HeatMap(data)
    ret.opts(opts.HeatMap(tools=['hover'], colorbar=True, toolbar='above'))
    ret.opts(xlabel='hop', ylabel='graph', shared_axes=False)

    return ret


        



