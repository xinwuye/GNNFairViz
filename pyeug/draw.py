from . import util
from .metrics import node_classification
import numpy as np
import holoviews as hv
from holoviews import opts, streams
# import os
# import torch
import numpy as np
# from sklearn.manifold import TSNE
# from holoviews.streams import Stream, param
# from holoviews import streams
# import datashader as ds
# import datashader.transfer_functions as tf
import pandas as pd
# from datashader.bundling import connect_edges, hammer_bundle
from holoviews.operation.datashader import datashade, bundle_graph
import panel as pn
from bokeh.models import CustomJSTickFormatter
# from tqdm import tqdm
from . import RangesetCategorical
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from bokeh.models import HoverTool
import matplotlib.colors as mcolors

hv.extension('bokeh')

PERPLEXITY = 15 # german
SEED = 42

def draw_graph_view(xy, source_id, target_id, layer, sens, 
                    sens_names, sens_name, bundle, sampled_alpha, colors):
    # print(layer)
    # print(xy)
    current_xy = xy[layer]
    x = current_xy[:,0]
    y = current_xy[:,1]
    node_indices = np.arange(0, len(x))
    # find the index of sens_name in sens_names
    sens_name_idx = []
    for sn in sens_name:
        sn_idx = sens_names.index(sn)
        sens_name_idx.append(sn_idx)

    tmp_sens_selected = sens[sens_name_idx]
    # Vectorize the function
    vectorized_concat = np.vectorize(util.concatenate_elements)
    # Apply the vectorized function across columns
    sens_selected = vectorized_concat(*tmp_sens_selected)

    # sens_name_idx = sens_names.index(sens_name)
    # sens_selected = sens[sens_name_idx]
    sens_selected_unique = np.unique(sens_selected)
    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
    nodes = hv.Nodes((x, y, node_indices, sens_selected, sampled_alpha), vdims=['Sens', 'sampled_alpha']) \
        .opts(opts.Nodes(alpha='sampled_alpha', nonselection_alpha=0))
    graph = hv.Graph(((source_id, target_id), nodes, ))
    graph.nodes.opts(opts.Nodes(color='Sens', alpha='sampled_alpha', cmap=colormap, size=3, line_color=None,))
    _ = graph.opts(cmap=colormap, node_size=4, edge_line_width=0.1,
                node_line_color='gray', node_color='Sens', node_alpha='sampled_alpha')
    if bundle:
        g = bundle_graph(graph)
    else:
        g = graph

    # draw poly
    poly_df = RangesetCategorical.compute_contours(colormap, current_xy, sens_selected, threshold=10)
    polys = hv.Polygons([{('x', 'y'): list(zip(poly_df.iloc[i]['xs'][0][0], poly_df.iloc[i]['ys'][0][0])), 'level': poly_df.iloc[i]['color']} for i in range(len(poly_df))], vdims='level')
    polys.opts(color='level', line_width=0, alpha=0.5)

    ret = (g * polys * g.nodes).opts(
        opts.Nodes(color='Sens', alpha='sampled_alpha', size=4, height=300, width=300, cmap=colors, 
            # legend_position='right', 
            show_legend=False,
            tools=['lasso_select', 'box_select']),
        opts.Graph(edge_line_width=0.1, inspection_policy='nodes', 
            edge_hover_line_color='green', node_hover_fill_color='red')
        ).opts(xlabel='x_graph_view', ylabel='y_graph_view', border=0)

    return ret


def draw_distribution(xy, layer, sens, sens_names, sens_name, colors, x_or_y):
    # sens_name_idx = sens_names.index(sens_name)
    # sens_selected = sens[sens_name_idx]

    # find the index of sens_name in sens_names
    sens_name_idx = []
    for sn in sens_name:
        sn_idx = sens_names.index(sn)
        sens_name_idx.append(sn_idx)

    tmp_sens_selected = sens[sens_name_idx]
    # Vectorize the function
    vectorized_concat = np.vectorize(util.concatenate_elements)
    # Apply the vectorized function across columns
    sens_selected = vectorized_concat(*tmp_sens_selected)

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


def draw_bar_metric_view(data, metric_name):
    # Creating a bar chart
    bar_chart = hv.Bars(data, 'Sensitive Subgroup', metric_name).opts(opts.Bars(xrotation=90))
    return bar_chart


def draw_heatmap_metric_view(data, xlabel, ylabel, title):
    """
    Creates a heatmap visualization using Holoviews.

    Parameters:
    data (list of tuples): Data for the heatmap, each tuple should contain (x_label, y_label, value).
    xlabel (str): Label for the x-axis.
    ylabel (str): Label for the y-axis.
    title (str): Title of the heatmap.

    Returns:
    hv.HeatMap: A Holoviews HeatMap object.
    """
    heatmap = hv.HeatMap(data).opts(
        tools=['hover'], 
        colorbar=True, 
        cmap='Viridis', 
        xlabel=xlabel, 
        ylabel=ylabel, 
        title=title
    )
    return heatmap


# Function to draw the correlation view selection (heatmap)
def draw_correlation_view_selection(correlations, p_vals, text, cmap_name='coolwarm'):
    # height = 100
    # width = height * len(correlations)
    cmap = plt.colormaps.get_cmap(cmap_name)
    custom_hover = HoverTool(tooltips=[("Feature", "@x0"), ("Correlation", "@Correlation")])

    rects, outlines, labels = [], [], []
    for i, (corr, p) in enumerate(zip(correlations, p_vals)):
        color = cmap((corr + 1) / 2)
        rects.append(hv.Rectangles([(i, 0, i+1, 1, corr)], vdims='Correlation').opts(fill_color=color))
        if p < 0.05:
            outlines.append(hv.Bounds((i, 0, i+1, 1)).opts(color='red', line_width=2))
        labels.append(hv.Text(i + 0.5, 0.5, text[i], halign='center', valign='center'))

    heatmap = hv.Overlay(rects + outlines + labels).opts(
        # opts.Rectangles(tools=[custom_hover], toolbar=None, active_tools=[], width=width, height=height),
        opts.Rectangles(tools=[custom_hover], toolbar=None, active_tools=[], ),
        opts.Bounds(line_dash='dotted'),
        opts.Text(fontsize=12),
        opts.Overlay(show_frame=False, yaxis=None, xaxis=None, show_legend=False)
    )

    return heatmap


# Function to draw the correlation view hex plot
def draw_correlation_view_hex(x, y, feat, pdd, gridsize=30):
    if x is None:
        index = 0
    else:
        index = int(x)  # Assuming x-coordinate corresponds to the index
    x_data = feat[index]
    y_data = pdd
    return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(gridsize=gridsize, tools=['hover'], colorbar=True))


def draw_correlation_view_violin(x, y, feat, pdd):
    if x is None:
        index = 0
    else:
        index = int(x)  # Assuming x-coordinate corresponds to the index
    x_data = feat[index]
    y_data = pdd
    data = pd.DataFrame({'feat': x_data, 'pdd': y_data})
    violin_list = [hv.Violin(data[data['feat'] == i], vdims='pdd') for i in data['feat'].unique()]
    violin_plot = hv.Overlay(violin_list)
    violin_plot.opts(opts.Violin(tools=['hover'], cmap='Category20'))

    return violin_plot


def draw_correlation_view_legend(cmap_name='coolwarm'):
    # Get the colormap
    cmap = plt.get_cmap(cmap_name)

    # Convert the colormap to a list of colors in hexadecimal format
    colors = [mcolors.rgb2hex(cmap(i)) for i in range(cmap.N)]

    # Define a 1D array
    data = np.linspace(-1, 1, 256)

    # Add an extra dimension to make it a 2D array
    data = data[None, :]

    # Create an Image with the 2D array and apply the colormap
    img = hv.Image(data, bounds=(-1,0,1,0.1)).opts(cmap=colors, colorbar=False, yaxis=None, 
                                                toolbar=None, framewise=True, title='',
                                                xlabel='Correlation')
    
    return img


