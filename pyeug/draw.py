from . import util
from .metrics import node_classification
import numpy as np
import holoviews as hv
from holoviews import opts, streams
# import os
import torch
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
from scipy.stats import kruskal, mannwhitneyu
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


def draw_density_view_scatter(graph_metrics):
    # graph_metrics is a list of tuples (num_nodes, density)
    # scatter_data = [(metric[0], metric[1]) for metric in graph_metrics]
    scatter_plot = hv.Scatter(graph_metrics, kdims=['Number of Nodes'], vdims=['Density'])
    return scatter_plot.opts(tools=['hover', 'tap'], active_tools=['tap'])


def draw_subgraph_view_heatmap(communities, index):
    # get the first element of list index if it is not empty
    if index:
        community = communities[index[0]]
        # Extract non-zero indices
        rows, cols = community.nonzero()

        # Prepare data for Holoviews
        # We create a list of tuples (i, j, 1) for each non-zero element
        data = [(rows[i], cols[i], 1) for i in range(len(rows))]

        # Define a Holoviews Dataset
        hv_dataset = hv.Dataset(data, ['i', 'j'], 'value')

        # Create a heatmap using the dataset
        # We use a colormap that maps 1 to black
        heatmap = hv.HeatMap(hv_dataset).opts(cmap=['black'], colorbar=False)
    else:
        heatmap = hv.HeatMap([])

    return heatmap


# def draw_attribute_view_violin(variable_data, feat_name, groups):
#     # If variable_data is a numpy array, convert it to a Pandas Series
#     if isinstance(variable_data, np.ndarray):
#         variable_data = pd.Series(variable_data, name=feat_name)
    
#     # Ensure groups is a Series and has the same index as variable_data
#     if isinstance(groups, np.ndarray):
#         groups = pd.Series(groups, name="Group")
    
#     # Combine the variable and groups into a single DataFrame
#     df = pd.concat([variable_data, groups], axis=1)
    
#     # Draw the violin plot for the specified feature, grouped by 'Group'
#     violin = hv.Violin(df, 'Group', feat_name).opts(toolbar=None, ylabel=feat_name)
    
#     return violin
def draw_attribute_view_violin(variable_data, feat_name, groups):
    if isinstance(variable_data, np.ndarray):
        variable_data = pd.Series(variable_data, name=feat_name)
    
    if isinstance(groups, np.ndarray):
        groups = pd.Series(groups, name="Group")
    
    df = pd.concat([variable_data, groups], axis=1)
    
    # Global statistical test (Kruskal-Wallis)
    kruskal_stat, kruskal_p = kruskal(*[group[feat_name].values for name, group in df.groupby("Group")])
    
    # Pairwise tests (Mann-Whitney U) for each group
    group_colors = {}
    for group_name in df['Group'].unique():
        group_data = df[df['Group'] == group_name][feat_name]
        other_data = df[df['Group'] != group_name][feat_name]
        
        stat, p = mannwhitneyu(group_data, other_data, alternative='two-sided')
        group_colors[group_name] = 'green' if p > 0.05 else 'red'  # Use 'red' to indicate differnt distribution

    # Create individual violin plots for each group and set colors
    violin_plots = []
    for group_name in df['Group'].unique():
        group_df = df[df['Group'] == group_name]
        color = group_colors[group_name]
        violin = hv.Violin(group_df, ('Group', 'Group'), feat_name).opts(violin_fill_color=color)
        violin_plots.append(violin)

    # Overlay the plots to create a combined plot with different colors
    violin_plot = hv.Overlay(violin_plots).opts(toolbar=None, ylabel=f"{feat_name} (p={kruskal_p:.3f})", 
                                                shared_axes=False)
    if kruskal_p <= 0.05:
        # Define a hook function to modify the plot
        def customize_plot(plot, element):
            plot.handles['yaxis'].axis_label_text_color = 'red'
        violin_plot.opts(hooks=[customize_plot])
    
    return violin_plot


def draw_attribute_view_bar(variable_data, feat_name, groups):
    # Ensure variable_data and groups are pandas Series with the same length
    if isinstance(variable_data, np.ndarray):
        variable_data = pd.Series(variable_data, name=feat_name)
    if isinstance(groups, np.ndarray):
        groups = pd.Series(groups, name="Group")
    
    # Combine into a DataFrame
    df = pd.concat([variable_data, groups], axis=1)
    
    # Count occurrences of each category within each group
    aggregated_df = df.groupby(['Group', feat_name]).size().reset_index(name='Count')
    
    # Create the bar chart
    bars = hv.Bars(aggregated_df, ['Group', feat_name], 'Count').opts(stacked=True, width=400, tools=['hover'],
                                                                      shared_axes=False,)
    
    return bars


def draw_attribute_view_sens(arr):
    """
    Draws a bar chart for the categorical values contained in a NumPy array using HoloViews.

    Parameters:
    - arr: np.array, an array consisting of categorical values.

    Returns:
    A HoloViews object that renders a bar chart visualizing the frequency of each category.
    """
    # Calculate the frequency of each unique category in the array
    unique, counts = np.unique(arr, return_counts=True)
    
    # Prepare the data for HoloViews in the form of a list of tuples (category, frequency)
    data = list(zip(unique, counts))
    
    # Create a Bar chart using HoloViews
    bars = hv.Bars(data, hv.Dimension('Category'), 'Frequency').opts(
        tools=['hover'],  # Enable hover tool for more interactive charts
        xlabel='Category', ylabel='Frequency',  # Label axes
        title='Categorical Value Frequencies',  # Chart title
        shared_axes=False, width=400
    )
    
    return bars