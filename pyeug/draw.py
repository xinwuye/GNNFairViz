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
from bokeh.models import HoverTool
# from tqdm import tqdm
from . import RangesetCategorical
from scipy.spatial.distance import cdist
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LinearSegmentedColormap
from bokeh.models import CategoricalColorMapper
from bokeh.models import CustomJS

hv.extension('bokeh')

PERPLEXITY = 15 # german
SEED = 42


# def draw_embedding_view(xy, layer, groups, sampled_alpha, colors, polygons_lst, max_edges_lst, threshold, index):
#     node_size = 3
#     current_xy = xy[layer]
#     x = current_xy[:, 0]
#     y = current_xy[:, 1]

#     sens_selected = groups
#     sens_selected_unique = np.unique(sens_selected)

#     colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
#     color_column = [colormap[group] for group in groups]

#     # if index is an empty list
#     if not index:
#         print('index is empty')
#         print(index)
#     else: 
#         print('index is not empty') 
#         print(index)

#     # Prepare DataFrame with color information
#     scatter_data = {'x': x, 'y': y, 'color': color_column, 'alpha': sampled_alpha}
#     df = pd.DataFrame(scatter_data)
#     scatter = hv.Scatter(df, kdims=['x', 'y'], vdims=['color', 'alpha']).opts(
#         size=node_size, 
#         color='color', 
#         alpha='alpha',  
#         line_color=None,
#         framewise=True,
#         hooks=[customize_unselected_glyph])

#     # Calculate and draw polygons (regions or clusters) as before
#     poly_df = RangesetCategorical.compute_contours(colormap, polygons_lst, max_edges_lst, threshold)
#     polys = hv.Polygons([{('x', 'y'): list(zip(poly_df.iloc[i]['xs'][0][0], poly_df.iloc[i]['ys'][0][0])), 
#                           'level': poly_df.iloc[i]['color']} for i in range(len(poly_df))], vdims='level')
#     polys.opts(color='level', line_width=0, alpha=0.3, framewise=True, active_tools=[], tools=[])

#     xmin, xmax = x.min(), x.max()
#     ymin, ymax = y.min(), y.max()

#     # Combine scatterplot and polygons
#     plot = (polys * scatter).opts( 
#         opts.Scatter(tools=['lasso_select', 'box_select'], 
#                     #  show_legend=True,
#                     #  legend_position='right',
#                      framewise=True,
#                      )
#     ).opts(
#         hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
#         show_legend=True,
#         border=0,
#         # show the toolbar at the top
#         toolbar='above',
#         framewise=True, 
#         # set x and y ranges according to x and y
#         xlim=(xmin - abs(xmin) * 0.1, xmax + abs(xmax) * 0.1), 
#         ylim=(ymin - abs(ymin) * 0.1, ymax + abs(ymax) * 0.1), 
#     ).opts(
#         opts.Polygons(active_tools=[], tools=[], framewise=True)
#     )
#     # plot = scatter

#     # # if index is an empty list
#     # if not index:
#     #     print('index is empty')
#     # else:
#     #     print('index is not empty') 

#     return plot


def draw_embedding_view_scatter(xy, layer, groups, alpha, colors):
    node_size = 5
    current_xy = xy[layer - 1]
    x = current_xy[:, 0]
    y = current_xy[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    sens_selected = groups
    sens_selected_unique = np.unique(sens_selected)

    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
    color_column = [colormap[group] for group in groups]

    # Prepare DataFrame with color information
    scatter_data = {'x': x, 'y': y, 'color': color_column, 'alpha': alpha}
    df = pd.DataFrame(scatter_data)
    scatter = hv.Scatter(df, kdims=['x', 'y'], vdims=['color', 'alpha']).opts(
        size=node_size, 
        color='color', 
        alpha='alpha',  
        line_color=None,
        framewise=True,
        # hooks=[customize_unselected_glyph]
    ).opts( 
        opts.Scatter(tools=['lasso_select', 'box_select', 'tap'],  
        )
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        border=0,
        # show the toolbar at the top
        toolbar='above',
        xlim=(xmin - abs(xmin) * 0.1, xmax + abs(xmax) * 0.1), 
        ylim=(ymin - abs(ymin) * 0.1, ymax + abs(ymax) * 0.1), 
        xaxis=None, yaxis=None, 
        shared_axes=False,
    )

    return scatter


def draw_embedding_view_polys(xy, layer, groups, colors, polygons_lst, max_edges_lst, threshold):
    current_xy = xy[layer - 1]
    x = current_xy[:, 0]
    y = current_xy[:, 1]

    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    sens_selected = groups
    sens_selected_unique = np.unique(sens_selected)

    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}

    # Calculate and draw polygons (regions or clusters) as before
    poly_df = RangesetCategorical.compute_contours(colormap, polygons_lst, max_edges_lst, threshold)
    polys = hv.Polygons([{('x', 'y'): list(zip(poly_df.iloc[i]['xs'][0][0], poly_df.iloc[i]['ys'][0][0])), 
                          'level': poly_df.iloc[i]['color']} for i in range(len(poly_df))], vdims='level')
    polys.opts(color='level', line_width=0, alpha=0.3, framewise=True, active_tools=[], tools=[]).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        border=0,
        # show the toolbar at the top
        toolbar='above',
        # set x and y ranges according to x and y
        xlim=(xmin - abs(xmin) * 0.1, xmax + abs(xmax) * 0.1), 
        ylim=(ymin - abs(ymin) * 0.1, ymax + abs(ymax) * 0.1), 
        xaxis=None, yaxis=None, 
        shared_axes=False,
    )

    return polys


# def draw_embedding_view_square(xy, layer, index):
#     if len(index) == 1:
#         current_xy = xy[layer]
#         x = current_xy[:, 0]
#         y = current_xy[:, 1]
#         x_selected = x[index]
#         y_selected = y[index]
#         square = hv.Rectangles([(x_selected[0]-0.05, y_selected[0]-0.05, x_selected[0]+0.05, y_selected[0]+0.05)])
#     else:
#         square = hv.Rectangles([])
#     return square.opts(
#         opts.Rectangles(fill_alpha=1, fill_color='red', line_color='red', line_width=2)
#     )

def draw_embedding_view_square(xy, layer, index):
    # Check if index is not empty
    if len(index) == 1:
        current_xy = xy[layer - 1]
        x = current_xy[:, 0]
        y = current_xy[:, 1]
        x_selected = x[index]
        y_selected = y[index]
        # Create a scatter plot for the selected points
        scatter = hv.Scatter((x_selected, y_selected))
    else:
        # If no points are selected, return an empty scatter plot
        scatter = hv.Scatter([]) 
    return scatter.opts(  
        opts.Scatter(size=5, color='black', line_color='black', fill_alpha=0, line_width=3, marker='s') 
    )


# def draw_distribution(xy, layer, sens, sens_names, sens_name, colors, x_or_y):
#     # sens_name_idx = sens_names.index(sens_name)
#     # sens_selected = sens[sens_name_idx]

#     # find the index of sens_name in sens_names
#     sens_name_idx = []
#     for sn in sens_name:
#         sn_idx = sens_names.index(sn)
#         sens_name_idx.append(sn_idx)

#     tmp_sens_selected = sens[sens_name_idx]
#     # Vectorize the function
#     vectorized_concat = np.vectorize(util.concatenate_elements)
#     # Apply the vectorized function across columns
#     sens_selected = vectorized_concat(*tmp_sens_selected)

#     sens_selected_unique = np.unique(sens_selected)
#     colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
#     current_xy = xy[layer]
#     x = current_xy[:,0]
#     y = current_xy[:,1]
#     node_indices = np.arange(0, len(x))
#     nodes = hv.Nodes((x, y, node_indices, sens_selected), vdims=['Sens'])
#     # draw distribution
#     dist = None
#     for ssu in sens_selected_unique:
#         points = nodes.select(Sens=ssu)
#         tmp_dist = hv.Distribution(points, kdims=[x_or_y]).opts(color=colormap[ssu])
#         if dist is None:
#             dist = tmp_dist
#             # ydist = tmp_ydist
#         else:
#             dist *= tmp_dist
#             # ydist *= tmp_ydist

#     if x_or_y == 'y':
#         return dist.opts(width=60, xaxis=None, yaxis=None, border=0)
#     else:
#         return dist.opts(height=60, xaxis=None, yaxis=None, border=0)


# def draw_legend(sens, sens_names, sens_name, colors):
#     # colors = hv.Cycle('Category20').values
#     # colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', '#4e7670']
#     # use point and text elements to draw the legend, which should correspond to unique values in sens[sen_name_idx] and node colors in graph.nodes
#     # get the unique values in sens[sen_name_idx]
#     sens_name_idx = sens_names.index(sens_name)
#     unique_sens = np.unique(sens[sens_name_idx])
#     n_unique_sens = len(unique_sens)
#     # draw the pointss vertically
#     x_legend = np.zeros(n_unique_sens)
#     y_legend = np.arange(0, n_unique_sens)
#     # use linspace to make the points evenly spaced
#     # y_legend = np.linspace(0, 0.1, len(unique_sens))
#     # reverse the order of the y_legend
#     y_legend = y_legend[::-1]
#     # draw the texts
#     texts = unique_sens
#     # draw the points
#     points = hv.Points((x_legend, y_legend, texts), vdims=['Sens']) \
#         .opts(opts.Points(size=10, color='Sens', cmap=colors, nonselection_alpha=0.1, height=30 * n_unique_sens, 
#             width=100, xaxis=None, yaxis=None, border=0, padding=0.5))
#     # draw the texts
#     for i, text in enumerate(texts):
#         points = points * hv.Text(x_legend[i] + 0.6, y_legend[i], text)
#     points.opts(show_legend=False, toolbar=None)

#     return points


def draw_embedding_view_legend(groups, colors):
    unique_sens = np.unique(groups)
    n_unique_sens = len(unique_sens)
    # draw the pointss vertically
    x_legend = np.zeros(n_unique_sens)
    y_legend = np.arange(0, n_unique_sens) / (n_unique_sens-1)
    # draw the texts
    texts = unique_sens
    colormap = {unique_sens[i]: colors[i] for i in range(len(unique_sens))}
    color_column = np.array([colormap[group] for group in unique_sens])
    # print(texts)
    # print(color_column)
    # draw the points
    points = hv.Points((x_legend, y_legend, color_column), vdims=['color']) \
        .opts(opts.Points(size=5, color='color', xaxis=None, yaxis=None))
    # draw the texts
    for i, text in enumerate(texts):
        points = points * hv.Text(x_legend[i] + 0.6, y_legend[i], text, halign='left', valign='center', fontsize=6)
    points.opts(show_legend=False, 
                toolbar=None,
                shared_axes=False,
                # x range
                xlim=(-1, 10),
                ylim=(-0.1, 0.1+1),
                hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                border=0
    )

    return points


# def draw_edge_legend():
#     # colors_edge is a dict: {edge_type: color}
#     colors_edge = {1: '#FE0000', 2: '#F4E0B9', 3: '#7D7463'}
#     # draw the edge legend
#     # use line and text elements to draw the legend
#     # draw the lines
#     # draw hv.Path for each line
#     labels = ['unfair', 'fair', 'unfair & fair']
#     paths = []
#     text = None
#     for i, edge_type in enumerate(colors_edge.keys()):
#         paths.append([(0, i, edge_type), (0.5, i, edge_type)])
#         if text is None:
#             text = hv.Text(0.6, i, labels[i])
#         else:
#             text = text * hv.Text(0.6, i, labels[i])
#     path = hv.Path(paths, vdims=['edge_type'])

#     ret = (path * text).opts(
#         opts.Path(color='edge_type', height=30 * len(colors_edge), width=100, cmap=colors_edge,
#             xaxis=None, yaxis=None),
#         opts.Text(text_font_size='10pt', text_align='left', text_baseline='middle', show_legend=False, toolbar=None)
#         ).opts(
#             border=0, padding=0.5, xlim=(-0.5, 3.5), ylim=(-1, len(colors_edge))
#         )

#     return ret


def draw_bar_metric_view(data, metric_name):
    # Creating a bar chart
    bar_chart = hv.Bars(data, 'Sensitive Subgroup', metric_name).opts(opts.Bars(xrotation=90)).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        fill_color='#717171'
    )
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
        title=title,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
    )
    return heatmap


# Function to draw the correlation view selection (heatmap)
def draw_correlation_view_selection(correlations, p_vals, text, x, y, cmap_name='coolwarm'):
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
    ).opts(
        # hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
    )

    return heatmap


# Function to draw the correlation view hex plot
# def draw_correlation_view_hex(x, y, feat, metric, selected_nodes, gridsize=30):
#     if x is None:
#         index = 0
#     else:
#         index = int(x)  # Assuming x-coordinate corresponds to the index
#     x_data = feat[index][selected_nodes]
#     y_data = metric[selected_nodes]
#     return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(gridsize=gridsize, tools=['hover'], colorbar=True)).opts(
#         hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
#     )


# def draw_correlation_view_hex(index, feat, metric, selected_nodes, gridsize=30):
#     x_data = feat[index][selected_nodes]
#     y_data = metric[selected_nodes]
#     return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(gridsize=gridsize, tools=['hover'], colorbar=True)).opts(
#         hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
#     )

# def draw_correlation_view_hex(index, feat, individual_bias_metrics, selected_nodes, gridsize=30):
#     x_data = feat[index][selected_nodes]
#     y_data = individual_bias_metrics[selected_nodes]
#     return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(gridsize=gridsize, tools=['hover'], colorbar=True)).opts(
#         hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
#         # set x label to '# Neighbors'
#         xlabel='# of Neighbors',
#         ylabel='Bias Contribution',
#         xrotation=90
#     )
def draw_correlation_view_hex(index, feat, individual_bias_metrics, gridsize=30):
    x_data = feat[index]
    y_data = individual_bias_metrics
    return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(gridsize=gridsize, 
                                                            tools=[], active_tools=[], 
                                                            colorbar=True)).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        # set x label to '# Neighbors'
        xlabel='# of Neighbors',
        ylabel='Bias Contribution',
        xrotation=90,
        framewise=True,
        shared_axes=False,
        alpha=1
    )


def draw_correlation_view_scatter(index, feat, individual_bias_metrics, groups, alpha, colors):
    node_size = 5
    # Data preparation
    x = feat[index]
    y = individual_bias_metrics

    sens_selected = groups
    sens_selected_unique = np.unique(sens_selected)

    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
    color_column = [colormap[group] for group in groups]

    # Prepare DataFrame with color information
    scatter_data = {'x': x, 'y': y, 'color': color_column, 'alpha': alpha}
    df = pd.DataFrame(scatter_data)
    scatter = hv.Scatter(df, kdims=['x', 'y'], vdims=['color', 'alpha']).opts(
        size=node_size, 
        color='color', 
        alpha='alpha',  
        line_color=None,
        framewise=True,
        # hooks=[customize_unselected_glyph]
    ).opts( 
        opts.Scatter(tools=['lasso_select', 'box_select', 'tap'],  
        )
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        border=0,
        # show the toolbar at the top
        # toolbar='above',
        xaxis=None, yaxis=None, 
        shared_axes=False,
    )

    return scatter


def draw_correlation_view_square(hop, feat, individual_bias_metrics, index):
    # Check if index is not empty
    if len(index) == 1:
        x = feat[hop]
        y = individual_bias_metrics
        x_selected = x[index]
        y_selected = y[index]
        # Create a scatter plot for the selected points
        scatter = hv.Scatter((x_selected, y_selected))
    else:
        # If no points are selected, return an empty scatter plot
        scatter = hv.Scatter([])
    return scatter.opts(
        opts.Scatter(size=5, color='red', line_color='black', fill_alpha=0, line_width=3, marker='s')
    )
    


def draw_correlation_view_violin(x, y, feat, metric, selected_nodes):
    if x is None:
        index = 0
    else:
        index = int(x)  # Assuming x-coordinate corresponds to the index
    x_data = feat[index][selected_nodes]
    y_data = metric[selected_nodes]
    data = pd.DataFrame({'feat': x_data, 'metric': y_data})
    violin_list = [hv.Violin(data[data['feat'] == i], vdims='metric') for i in data['feat'].unique()]
    violin_plot = hv.Overlay(violin_list)
    violin_plot.opts(opts.Violin(tools=['hover'], cmap='Category20')).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
    )

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
                                                xlabel='Correlation',
                                                hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)])
    
    return img


def draw_density_view_scatter(graph_metrics):
    # graph_metrics is a list of tuples (num_nodes, density)
    # scatter_data = [(metric[0], metric[1]) for metric in graph_metrics]
    scatter_plot = hv.Scatter(graph_metrics, kdims=['# of Nodes'], vdims=['Density'])
    return scatter_plot.opts(
                             tools=['hover', 'tap', ], 
                             active_tools=['tap'],
                             hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                             framewise=True,
                             # set the color to #717171
                             color='#717171',
                             size=5
                             )


# def draw_subgraph_view_heatmap(communities, index):
#     # get the first element of list index if it is not empty
#     if index:
#         community = communities[index[0]]
#         # Extract non-zero indices
#         rows, cols = community.nonzero()

#         # Prepare data for Holoviews
#         # We create a list of tuples (i, j, 1) for each non-zero element
#         data = [(rows[i], cols[i], 1) for i in range(len(rows))]

#         # Define a Holoviews Dataset
#         hv_dataset = hv.Dataset(data, ['i', 'j'], 'value')

#         # Create a heatmap using the dataset
#         # We use a colormap that maps 1 to black
#         heatmap = hv.HeatMap(hv_dataset).opts(cmap=['black'], colorbar=False)
#     else:
#         heatmap = hv.HeatMap([])

#     return heatmap
def draw_subgraph_view_heatmap(selected_nodes, adj):
    colors = ["white", "#717171"]  # Colors from 0 to 1
    cmap = LinearSegmentedColormap.from_list("custom_cmap", colors, N=2)
    if len(selected_nodes) != adj.shape[0]:
        # Extract non-zero indices
        rows, cols = adj[selected_nodes][: , selected_nodes].nonzero()

        # Prepare data for Holoviews
        # We create a list of tuples (i, j, 1) for each non-zero element
        data = [(rows[i], cols[i], 1) for i in range(len(rows))]

        # Define a Holoviews Dataset
        hv_dataset = hv.Dataset(data, ['i', 'j'], 'value')
        # Create a heatmap using the dataset
        heatmap = hv.HeatMap(hv_dataset)
    else:
        heatmap = hv.HeatMap([])

    return heatmap.opts(hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                        shared_axes=False,
                        xaxis=None,
                        yaxis=None,
                        framewise=True,
                        border=0,
                        colorbar=False,
                        # set the color to #717171
                        cmap=cmap).redim.range(x=(0, len(selected_nodes)), y=(0, len(selected_nodes)))


# Define a hook function to modify the plot
def ylabel_red_hook(plot, element):
    plot.handles['yaxis'].axis_label_text_color = 'red'

    
def ylabel_black_hook(plot, element):
    plot.handles['yaxis'].axis_label_text_color = 'black'


def draw_attribute_view_violin(variable_data, feat_name, groups, selected_nodes):
    # If variable_data is a numpy array, convert it to a Pandas Series
    if isinstance(variable_data, np.ndarray):
        variable_data = pd.Series(variable_data, name=feat_name)
    
    # Ensure groups is a Series and has the same index as variable_data
    if isinstance(groups, np.ndarray):
        groups = pd.Series(groups, name="Group")
    
    # Combine the variable and groups into a single DataFrame
    df = pd.concat([variable_data, groups], axis=1)
    df = df.iloc[selected_nodes]
    # print(df)
    
    # Draw the violin plot for the specified feature, grouped by 'Group'
    violin = hv.Violin(df, 'Group', feat_name).opts(ylabel='Attribute',
                                                    xlabel='Sensitive Subgroup',
                                                    hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                                                    shared_axes=False,
                                                    violin_fill_color='#717171',
                                                    xrotation=90,
                                                    framewise=True,)
    
    return violin


# def draw_attribute_view_violin(variable_data, feat_name, groups, selected_nodes):
#     if isinstance(variable_data, np.ndarray):
#         variable_data = pd.Series(variable_data, name=feat_name)
    
#     if isinstance(groups, np.ndarray):
#         groups = pd.Series(groups, name="Group")
    
#     df = pd.concat([variable_data, groups], axis=1)
#     df = df.iloc[selected_nodes]
    
#     # Global statistical test (Kruskal-Wallis)
#     kruskal_stat, kruskal_p = kruskal(*[group[feat_name].values for name, group in df.groupby("Group")])
    
#     # Pairwise tests (Mann-Whitney U) for each group
#     group_colors = {}
#     for group_name in df['Group'].unique():
#         group_data = df[df['Group'] == group_name][feat_name]
#         other_data = df[df['Group'] != group_name][feat_name]
        
#         stat, p = mannwhitneyu(group_data, other_data, alternative='two-sided')
#         group_colors[group_name] = 'green' if p > 0.05 else 'red'  # Use 'red' to indicate differnt distribution

#     # Create individual violin plots for each group and set colors
#     violin_plots = []
#     for group_name in df['Group'].unique():
#         group_df = df[df['Group'] == group_name]
#         color = group_colors[group_name]
#         violin = hv.Violin(group_df, ('Group', 'Group'), feat_name).opts(violin_fill_color=color)
#         violin_plots.append(violin)

#     # Overlay the plots to create a combined plot with different colors
#     violin_plot = hv.Overlay(violin_plots).opts(toolbar=None, ylabel=f"{feat_name} (p={kruskal_p:.3f})", 
#                                                 shared_axes=False)
#     if kruskal_p <= 0.05:
#         violin_plot.opts(hooks=[ylabel_red_hook])
#     else:
#         violin_plot.opts(hooks=[ylabel_black_hook])
    
#     return violin_plot.opts(hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)])


# def draw_attribute_view_bar(variable_data, feat_name, groups, selected_nodes):
#     # Ensure variable_data and groups are pandas Series with the same length
#     if isinstance(variable_data, np.ndarray):
#         variable_data = pd.Series(variable_data, name=feat_name)
#     if isinstance(groups, np.ndarray):
#         groups = pd.Series(groups, name="Group")

#     variable_data = variable_data.iloc[selected_nodes]
#     groups = groups.iloc[selected_nodes]

#     # Step 1: Construct a Contingency Table
#     contingency_table = pd.crosstab(variable_data, groups)

#     # Step 2: Perform the Chi-Square Test
#     chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)

#     # Pairwise tests (Chi-Square) for each group
#     group_colors = {}
#     for group_name in groups.unique():
#         group_data = contingency_table[group_name]
#         other_data = contingency_table.drop(columns=group_name)
#         stat, p, dof, expected = chi2_contingency(pd.concat([group_data, other_data], axis=1))
#         group_colors[group_name] = 'green' if p > 0.05 else 'red'
    
#     # Combine into a DataFrame
#     df = pd.concat([variable_data, groups], axis=1)
    
#     # Count occurrences of each category within each group
#     aggregated_df = df.groupby(['Group', feat_name]).size().reset_index(name='Count')
    
#     # Create the bar chart
#     bars = hv.Bars(aggregated_df, ['Group', feat_name], 'Count').opts(stacked=False, width=400, tools=['hover'],
#                                                                       xlabel='Group', ylabel='Count',
#                                                                       title=f'{feat_name} (p={chi2_p:.3f})',
#                                                                       shared_axes=False,
#                                                                       color=hv.dim('Group').categorize(group_colors),
#                                                                       hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
#                                                                       xrotation=90)
#     if chi2_p <= 0.05:
#         bars.opts(hooks=[ylabel_red_hook])
#     else:
#         bars.opts(hooks=[ylabel_black_hook])
#     return bars


def draw_attribute_view_bar(variable_data, feat_name, groups, selected_nodes):
    # Ensure variable_data and groups are pandas Series with the same length
    if isinstance(variable_data, np.ndarray):
        variable_data = pd.Series(variable_data, name=feat_name)
    if isinstance(groups, np.ndarray):
        groups = pd.Series(groups, name="Group")

    variable_data = variable_data.iloc[selected_nodes]
    # convert variable_data to string
    variable_data = variable_data.astype(str)
    groups = groups.iloc[selected_nodes]

    # create a column count with the same length as variable_data and all values are 1
    count = np.ones(len(variable_data))
    # create a DataFrame with variable_data, groups, and count
    df = pd.DataFrame({feat_name: variable_data, 'Group': groups, 'Count': count})

    # Step 1: Construct a Contingency Table
    # contingency_table = pd.crosstab(variable_data, groups)

    # Step 2: Perform the Chi-Square Test
    # chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
    
    # Combine into a DataFrame
    # df = pd.concat([variable_data, groups], axis=1)
    
    # # Count occurrences of each category within each group
    # aggregated_df = df.groupby(['Group', feat_name]).size().reset_index(name='Count')

    # # Create the bar chart
    # bars = hv.Bars(aggregated_df, ['Group', feat_name], 'Count').opts(
    bars = hv.Bars(df, ['Group', feat_name], 'Count').aggregate(function=np.sum).opts(
        stacked=False, tools=['hover'],
        xlabel='Sensitive Subgroup', ylabel='Count',
    #   title=f'{feat_name} (p={chi2_p:.3f})',
        shared_axes=False,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        invert_axes=True,
        multi_level=False,
        # color='',
        cmap=['white', '#717171'],
        # put legend on the right
        legend_position='right',
        # put toolbar on the top
        toolbar='above',
    )

    return bars


def draw_attribute_view_overview(feat, groups, columns_categorical, selected_nodes, 
                                 individual_bias_metrics, x, y, contributions, hw_ratio):
    print(contributions)
    n_all_nodes = len(feat)
    # print(groups)
    groups = pd.Series(groups, name="Group")
    bias_indicators, overall_bias_indicators, ns, all_ns, all_unique_groups = util.analyze_bias(feat, groups, columns_categorical, selected_nodes)
    # # k being the sum of overall_bias_indicator
    # k = len(overall_bias_indicators) - overall_bias_indicators.sum()
    # print(ns)
    # ns being the number of each unique value in 
    m = len(ns)
    # print(ns)
    n = len(overall_bias_indicators)

    xmin = -n_all_nodes / 10 - 5 -2 
    xmax = n_all_nodes * 1.1
    ymin = 0
    ymax = n+1

    # Prepare the data for Rectangles
    rect_data = []
    x0 = 0
    for j in range(m):
        # print(x0)
        width = ns[j]
        column = bias_indicators[:, j]
        # print([(x0, i, x0 + width, i+1, column[i]) for i in range(n)][-4])
        rect_data.extend([(x0, i, x0 + width, i+1, column[i], i) for i in range(n)])
        x0 += all_ns[j]

    rect_data = pd.DataFrame(rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Color', 'ID'])        

    # Calculate the center of each column
    column_centers = [sum(all_ns[:i+1]) - all_ns[i]/2 for i in range(m)]
    # create yticks using column_centers and all_unique_groups
    yticks = [(c, all_unique_groups[i]) for i, c in enumerate(column_centers)]

    # Create the Rectangles plot
    plot = hv.Rectangles(rect_data, vdims=['Color', 'ID']).opts(
        opts.Rectangles(
                        # tools=['tap'], active_tools=['tap'],
                        color=hv.dim('Color').categorize(
                            {False: '#87CEEB', True: '#FF6347'}),
                        # yformatter='%.0f',  # Show integers on y-axis
                        # yticks=list(range(n)),  # Set y-axis ticks
                        # yaxis=None,  # Hide y-axis
                        xaxis=None,  # Hide x-axis
                        xlabel='Sensitive Subgroup',  # X-axis label
                        yticks=yticks,
                        line_width=0.1,
                        # alpha=0.3,
                        # xrotation=90
                        ))  # Column tick labels
    
    # glyph plot
    r_sector1 = 0.48
    r_ellipse1 = r_sector1 * 1.5
    r_ellipse2 = r_sector1
    r_sector2 = r_sector1 / 2 

    circle_x = -n_all_nodes / 20 - 2

    # calculate the proportion of x range / y range
    x_y_ratio = (xmax - xmin) / (ymax - ymin) / hw_ratio

    # # Calculate correlations and p-values
    feat_np = feat.to_numpy()
    # Placeholder lists for correlations and p-values
    correlations = []
    p_values = []

    # Iterate over all features in feat_np using their index to maintain order
    for index, is_categorical in enumerate(columns_categorical):
        # Extract the feature column
        feature_column = feat_np[:, index]
        
        # Check if the feature is continuous or categorical and calculate accordingly
        if is_categorical:
            # Point-Biserial correlation for categorical features
            correlation, p_val = stats.pointbiserialr(feature_column, individual_bias_metrics)
        else:
            # Pearson correlation for continuous features
            correlation, p_val = stats.pearsonr(feature_column, individual_bias_metrics)
        
        # Append the results to the lists
        correlations.append(abs(correlation))
        p_values.append(p_val)

    # contributions is a 1d np array, get the max absolute value
    max_contrib = np.max(np.abs(contributions))
    # Normalize contributions to [-1, 1]
    normalized_contributions = contributions / max_contrib
    sector_pointes1_angles = normalized_contributions * 180

    n = bias_indicators.shape[0]

    ellipse_data = []
    for i in range(n):
        # sector_pointes1: contributions
        end_angle = sector_pointes1_angles[i]
        sector_points1 = create_sector(center=(circle_x, i + 0.5),
                                        radius=r_sector1,
                                        x_y_ratio = x_y_ratio,
                                        start_angle=90,
                                        end_angle=90-end_angle,)
        sector_data1 = {('x', 'y'): sector_points1, 'color': '#717171'}
        ellipse_data.append(sector_data1)

        # ellipse1: overall bias indicators
        # # Determine color based on row index relative to k
        # color = 'green' if i < k else 'red'
        color = '#FF6347' if overall_bias_indicators[i] else '#87CEEB'
        e = hv.Ellipse(circle_x, i + 0.5, (x_y_ratio * r_ellipse1, r_ellipse1))
        e_data = {('x', 'y'): e.array(), 'color': color}
        ellipse_data.append(e_data)

        # ellipse2: just white
        e = hv.Ellipse(circle_x, i + 0.5, (x_y_ratio * r_ellipse2, r_ellipse2))
        e_data = {('x', 'y'): e.array(), 'color': 'white'}
        ellipse_data.append(e_data)

        # sector_points2: angle for correlations, color for p-values
        p_val = p_values[i]
        color = '#FF6347' if p_val < 0.05 else '#87CEEB'
        correlation = correlations[i]
        end_angle = 360 * correlation
        sector_points2 = create_sector(center=(circle_x, i + 0.5), 
                                       radius=r_sector2,
                                       x_y_ratio = x_y_ratio,
                                       start_angle=90, 
                                       end_angle=90-end_angle,)
        sector_data2 = {('x', 'y'): sector_points2, 'color': color}
        ellipse_data.append(sector_data2)

    glyph_plot = hv.Polygons(ellipse_data, vdims='color').opts(
        line_width=0,
        color='color',
    )

    # Prepare data for transparent rectangles with black strokes
    transparent_rect_data = []
    x0 = 0  # Starting x-coordinate
    for width in all_ns:
        # Add a rectangle for each group, transparent fill and black stroke
        transparent_rect_data.append((x0, 0, x0 + width, n, 'black'))
        x0 += width

    # Convert transparent rectangle data into a DataFrame
    transparent_rect_df = pd.DataFrame(transparent_rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Line_Color'])

    # Create Transparent Rectangles plot
    transparent_rectangles_plot = hv.Rectangles(transparent_rect_df, vdims=['Line_Color']).opts(
        fill_alpha=0,  # Set fill color to transparent
        line_color=hv.dim('Line_Color'),
        line_width=1,
        tools=[],
    )
    
    # Overlay Transparent Rectangles onto the existing combined plot with rectangles
    final_combined_plot = (plot * glyph_plot * transparent_rectangles_plot).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        invert_axes=True,
        framewise=True,
        shared_axes=False,
    ).redim.range(x=(-n_all_nodes / 10 - 5-2, n_all_nodes*1.1), y=(0, n+1))
    
    # Return the final combined plot
    # return final_combined_plot

    if x:
        if 0 <= x <= n:
            # Draw an arrow pointing downwards at the top of the plot
            # Since the plot is inverted, the top is actually on the right, before inverting
            arrow_y_position = n_all_nodes*1.02  # Adjust this value as needed to place the arrow correctly
            arrow = hv.Arrow(int(x)+0.5, arrow_y_position, direction='v', arrowstyle='-|>')

            # Overlay the Arrow on the Final Combined Plot
            final_plot_with_arrow = final_combined_plot * arrow
        else:
            final_plot_with_arrow = final_combined_plot * hv.Arrow(np.nan, np.nan)
    else:
        final_plot_with_arrow = final_combined_plot

    # Return the final plot with the arrow
    return final_plot_with_arrow


# def draw_attribute_view_overview(feat, groups, columns_categorical, selected_nodes, individual_bias_metrics, x, y):
#     n_all_nodes = len(feat)
#     # print(groups)
#     groups = pd.Series(groups, name="Group")
#     bias_indicators, overall_bias_indicators, ns, all_ns, all_unique_groups = util.analyze_bias(feat, groups, columns_categorical, selected_nodes)
#     # # k being the sum of overall_bias_indicator
#     # k = len(overall_bias_indicators) - overall_bias_indicators.sum()
#     # print(ns)
#     # ns being the number of each unique value in 
#     m = len(ns)
#     # print(ns)
#     n = len(overall_bias_indicators)
#     # Prepare the data for Rectangles
#     rect_data = []
#     x0 = 0
#     for j in range(m):
#         # print(x0)
#         width = ns[j]
#         column = bias_indicators[:, j]
#         # print([(x0, i, x0 + width, i+1, column[i]) for i in range(n)][-4])
#         rect_data.extend([(x0, i, x0 + width, i+1, column[i], i) for i in range(n)])
#         x0 += all_ns[j]

#     rect_data = pd.DataFrame(rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Color', 'ID'])        

#     # Calculate the center of each column
#     column_centers = [sum(all_ns[:i+1]) - all_ns[i]/2 for i in range(m)]
#     # create yticks using column_centers and all_unique_groups
#     yticks = [(c, all_unique_groups[i]) for i, c in enumerate(column_centers)]

#     # Create the Rectangles plot
#     plot = hv.Rectangles(rect_data, vdims=['Color', 'ID']).opts(
#         opts.Rectangles(
#                         # tools=['tap'], active_tools=['tap'],
#                         color=hv.dim('Color').categorize(
#                             {False: '#87CEEB', True: '#FF6347'}),
#                         # yformatter='%.0f',  # Show integers on y-axis
#                         # yticks=list(range(n)),  # Set y-axis ticks
#                         # yaxis=None,  # Hide y-axis
#                         xaxis=None,  # Hide x-axis
#                         xlabel='Sensitive Subgroup',  # X-axis label
#                         yticks=yticks,
#                         line_width=0.1,
#                         # alpha=0.3,
#                         # xrotation=90
#                         ))  # Column tick labels

#    # Step 1: Correctly Generate Circle Data with Accurate Color Assignment
#     # Calculate the x-coordinate for circle centers (to the left of the first column)
#     # circle_x = -ns[0] / 2
#     circle_x = -n_all_nodes / 20 - 2
#     n = bias_indicators.shape[0]
#     circle_data_corrected = []
#     for i in range(n):
#         # # Determine color based on row index relative to k
#         # color = 'green' if i < k else 'red'
#         color = '#FF6347' if overall_bias_indicators[i] else '#87CEEB'
#         # Calculate y-coordinate for the center of the circle (midpoint of the rectangle's height)
#         y_center = i + 0.5
#         # Append circle information (x-coordinate, y-coordinate, and color)
#         circle_data_corrected.append((circle_x, y_center, color, i))

#     circle_data_corrected = pd.DataFrame(circle_data_corrected, columns=['x', 'y', 'Color', 'ID'])

#     # Step 2: Create Corrected Circles Plot
#     circles_plot_corrected = hv.Scatter(circle_data_corrected, ['x', 'y'], ['Color', 'ID']).opts(
#         color='Color', marker='circle', size=1,
#         tools=[],  # No interactive tools needed for circles
#         legend_position='top_right')

#     # Step 3: Overlay Corrected Circles Plot onto the Existing Rectangles Plot
#     combined_plot_corrected = plot * circles_plot_corrected

#     # return combined_plot_corrected.opts(hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)])

#     # # Calculate correlations and p-values
#     feat_np = feat.to_numpy()
#     # Placeholder lists for correlations and p-values
#     correlations = []
#     p_values = []

#     # Iterate over all features in feat_np using their index to maintain order
#     for index, is_categorical in enumerate(columns_categorical):
#         # Extract the feature column
#         feature_column = feat_np[:, index]
        
#         # Check if the feature is continuous or categorical and calculate accordingly
#         if is_categorical:
#             # Point-Biserial correlation for categorical features
#             correlation, p_val = stats.pointbiserialr(feature_column, individual_bias_metrics)
#         else:
#             # Pearson correlation for continuous features
#             correlation, p_val = stats.pearsonr(feature_column, individual_bias_metrics)
        
#         # Append the results to the lists
#         correlations.append(correlation)
#         p_values.append(p_val)

#     # Calculate the x-coordinate for the rectangles (to the left of the circles)
#     square_x = -n_all_nodes / 10 - 5  # This will need to be adjusted to prevent overlap with circles

#     # Prepare Rectangle Data to mimic Squares
#     rectangle_data = []
#     cmap = plt.get_cmap('coolwarm')
#     for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
#         # Convert the RGBA color from the colormap to a hexadecimal color
#         fill_color = mcolors.to_hex(cmap((corr + 1) / 2)) if not np.isnan(corr) else 'white'
#         line_color = 'red' if p_val < 0.05 else 'black'  # No line color if p-value is not significant

#         # Append rectangle data
#         rectangle_data.append((square_x, i, square_x + 10, i + 1, fill_color, line_color, corr, i))

#     # After the loop, convert the list of tuples to a DataFrame
#     rectangle_df = pd.DataFrame(rectangle_data, columns=['x0', 'y0', 'x1', 'y1', 'Fill_Color', 'Line_Color', 'Correlation', 'ID'])

#     # Create the Rectangles plot to mimic Squares
#     rectangles_plot = hv.Rectangles(rectangle_df, vdims=['Fill_Color', 'Line_Color', 'ID']).opts(
#         fill_color=hv.dim('Fill_Color'),  # Fill rectangles with the specified color
#         line_color=hv.dim('Line_Color'),  # Add a border around the rectangles
#         line_width=0.5,
#         # fill_alpha=0.5  # Adjust transparency as needed
#     )
    
#     # Overlay Rectangles Plot onto the existing combined plot with circles
#     combined_plot_with_rectangles = (combined_plot_corrected * rectangles_plot)
    
#     # Return the updated plot with rectangles as squares
#     # return combined_plot_with_rectangles

#     # Prepare data for transparent rectangles with black strokes
#     transparent_rect_data = []
#     x0 = 0  # Starting x-coordinate
#     for width in all_ns:
#         # Add a rectangle for each group, transparent fill and black stroke
#         transparent_rect_data.append((x0, 0, x0 + width, n, 'black'))
#         x0 += width

#     # Convert transparent rectangle data into a DataFrame
#     transparent_rect_df = pd.DataFrame(transparent_rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Line_Color'])

#     # Create Transparent Rectangles plot
#     transparent_rectangles_plot = hv.Rectangles(transparent_rect_df, vdims=['Line_Color']).opts(
#         fill_alpha=0,  # Set fill color to transparent
#         line_color=hv.dim('Line_Color'),
#         line_width=1,
#         tools=[],
#     )
    
#     # Overlay Transparent Rectangles onto the existing combined plot with rectangles
#     final_combined_plot = (combined_plot_with_rectangles * transparent_rectangles_plot).opts(
#         hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
#         invert_axes=True,
#         framewise=True,
#         shared_axes=False,
#     ).redim.range(x=(-n_all_nodes / 10 - 5-2, n_all_nodes*1.1), y=(0, n+1))
    
#     # Return the final combined plot
#     # return final_combined_plot

#     if x:
#         if 0 <= x <= n:
#             # Draw an arrow pointing downwards at the top of the plot
#             # Since the plot is inverted, the top is actually on the right, before inverting
#             arrow_y_position = n_all_nodes*1.02  # Adjust this value as needed to place the arrow correctly
#             arrow = hv.Arrow(int(x)+0.5, arrow_y_position, direction='v', arrowstyle='-|>')

#             # Overlay the Arrow on the Final Combined Plot
#             final_plot_with_arrow = final_combined_plot * arrow
#         else:
#             final_plot_with_arrow = final_combined_plot * hv.Arrow(np.nan, np.nan)
#     else:
#         final_plot_with_arrow = final_combined_plot

#     # Return the final plot with the arrow
#     return final_plot_with_arrow


def draw_attribute_view_correlation_violin(feat, individual_bias_metrics, selected_nodes):
    x_data = feat[selected_nodes]
    y_data = individual_bias_metrics[selected_nodes]
    data = pd.DataFrame({'feat': x_data, 'metric': y_data})
    violin_plot = hv.Violin(data, 'feat', 'metric').opts(
        xlabel='Atribute', ylabel='Bias Contribution').opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        violin_fill_color='#717171',
        framewise=True,
    )

    return violin_plot


def draw_attribute_view_correlation_hex(feat, individual_bias_metrics, selected_nodes, gridsize=30):
    x_data = feat[selected_nodes]
    y_data = individual_bias_metrics[selected_nodes]
    return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(
        gridsize=gridsize, 
        tools=['hover'], 
        colorbar=True)).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        shared_axes=False,
        xlabel='Attribute',
        ylabel='Bias Contribution',
        xrotation=90
    )


# def draw_fairness_metric_view_bar(labels, groups, selected_nodes, predictions): 
#     sens = groups
#     mask = selected_nodes
#     predictions = predictions[mask]
#     labels = labels[mask]
#     sens = sens[mask]
#     labeled_mask = labels != -1
#     labeled_predictions = predictions[labeled_mask]
#     labeled_labels = labels[labeled_mask]
#     labeled_sens = sens[labeled_mask]

#     letter_values = {
#         ('SP', 'STD'): [node_classification.delta_std_sp(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#         ('SP', 'MAX'): [node_classification.delta_max_sp(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#         ('EOP', 'STD'): [node_classification.delta_std_eop(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#         ('EOP', 'MAX'): [node_classification.delta_max_eop(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#         ('EOD', 'STD'): [node_classification.delta_std_eod(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#         ('EOD', 'MAX'): [node_classification.delta_max_eod(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#         ('Acc', 'STD'): [node_classification.delta_std_acc(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#         ('Acc', 'MAX'): [node_classification.delta_max_acc(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#     }

#     data = {
#         'Metric Type': [],
#         'Measurement Method': [],
#         'Value': [],
#         'Upper Bound': [],
#     }
#     label_data = {
#         'x': [],
#         'y': [],
#         'text': [],
#     }

#     for (metric_type, measurement_method), [value, bounds] in letter_values.items():
#         data['Metric Type'].append(metric_type)
#         data['Measurement Method'].append(measurement_method)
#         data['Value'].append(value)
#         data['Upper Bound'].append(bounds[1])

#     df = pd.DataFrame(data)

#     range_bars = hv.Bars(df, ['Metric Type', 'Measurement Method'], 'Upper Bound').opts(
#         hv.opts.Bars(fill_alpha=0, line_dash='dashed', line_width=1.5, stacked=False, invert_axes=True)
#     )
#     value_bars = hv.Bars(df, ['Metric Type', 'Measurement Method'], 'Value').opts(
#         hv.opts.Bars(fill_color='#717171', stacked=False, invert_axes=True, tools=['hover', 'tap'])
#     )

#     composite_chart = (range_bars * value_bars).opts(
#         hv.opts.Overlay(show_legend=False),
#     ).opts(
#         hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
#         xlabel='Fairness Metric',
#         ylabel='Value',
#         ylim=(0, 1.05),
#         xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
#         shared_axes=False,
#     )

#     return composite_chart


def draw_fairness_metric_view_value_bar(labels, groups, selected_nodes, predictions): 
    sens = groups
    mask = selected_nodes
    predictions = predictions[mask]
    labels = labels[mask]
    sens = sens[mask]
    labeled_mask = labels != -1
    labeled_predictions = predictions[labeled_mask]
    labeled_labels = labels[labeled_mask]
    labeled_sens = sens[labeled_mask]

    letter_values = {
        ('SP', 'STD'): [node_classification.delta_std_sp(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
        ('SP', 'MAX'): [node_classification.delta_max_sp(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
        ('EOP', 'STD'): [node_classification.delta_std_eop(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
        ('EOP', 'MAX'): [node_classification.delta_max_eop(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
        ('EOD', 'STD'): [node_classification.delta_std_eod(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
        ('EOD', 'MAX'): [node_classification.delta_max_eod(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
        ('Acc', 'STD'): [node_classification.delta_std_acc(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
        ('Acc', 'MAX'): [node_classification.delta_max_acc(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
    }

    data = {
        'Metric Type': [],
        'Measurement Method': [],
        'Value': [],
        'Upper Bound': [],
    }

    for (metric_type, measurement_method), [value, bounds] in letter_values.items():
        data['Metric Type'].append(metric_type)
        data['Measurement Method'].append(measurement_method)
        data['Value'].append(value)
        data['Upper Bound'].append(bounds[1])

    df = pd.DataFrame(data)

    custom_hover = HoverTool(tooltips=[('Value', '@{Value}')])

    value_bars = hv.Bars(df, ['Metric Type', 'Measurement Method'], 'Value').opts(
        hv.opts.Bars(fill_color='#717171', stacked=False, invert_axes=True, tools=[custom_hover, 'tap']) 
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        xlabel='Fairness Metric',
        ylabel='Value',
        ylim=(0, 1.05),
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        shared_axes=False,
        show_legend=False
    )

    return value_bars


def draw_fairness_metric_view_range_bar(): 
    letter_values = {
        ('SP', 'STD'): (0.0, 1.0),
        ('SP', 'MAX'): (0.0, 1.0),
        ('EOP', 'STD'): (0.0, 1.0),
        ('EOP', 'MAX'): (0.0, 1.0),
        ('EOD', 'STD'): (0.0, 1.0),
        ('EOD', 'MAX'): (0.0, 1.0),
        ('Acc', 'STD'): (0.0, 1.0),
        ('Acc', 'MAX'): (0.0, 1.0),
    }

    data = {
        'Metric Type': [],
        'Measurement Method': [],
        'Upper Bound': [],
    }

    for (metric_type, measurement_method), bounds in letter_values.items():
        data['Metric Type'].append(metric_type)
        data['Measurement Method'].append(measurement_method)
        data['Upper Bound'].append(bounds[1])

    df = pd.DataFrame(data)

    range_bars = hv.Bars(df, ['Metric Type', 'Measurement Method'], 'Upper Bound').opts(
        hv.opts.Bars(fill_alpha=0, line_width=1.5, stacked=False, invert_axes=True)
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        xlabel='Fairness Metric',
        ylabel='Value',
        ylim=(0, 1.05),
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        shared_axes=False,
        show_legend=False
    )

    return range_bars


def draw_fairness_metric_view_detail(metric_name, selected_nodes, groups, labels, predictions, eod_radio):
    mask = selected_nodes
    predictions = predictions[mask]
    labels = labels[mask]
    sens = groups[mask]

    labeled_mask = labels != -1
    predictions = predictions[labeled_mask] 
    labels = labels[labeled_mask]
    groups = sens[labeled_mask]
    unique_labels = np.unique(labels) 
    unique_groups = np.unique(groups) 

    if metric_name == 0 or metric_name == 4:
        heatmap_data = []

        for label in unique_labels:
            for group in unique_groups:
                group_indices = np.where(groups == group)
                group_predictions = predictions[group_indices]

                prob = np.mean(group_predictions == label)
                heatmap_data.append((str(int(label)), group, prob))

        chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'P(Pred.=Label)')
    elif metric_name == 1 or metric_name == 5:
        heatmap_data = []
        
        for label in unique_labels:
            for group in unique_groups:
                group_indices = np.where(groups == group)
                group_pred = predictions[group_indices]
                group_labels = labels[group_indices]

                tp = np.sum((group_pred == label) & (group_labels == label))
                fn = np.sum((group_pred != label) & (group_labels == label))
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

                heatmap_data.append((str(int(label)), group, tpr))
        
        chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'TPR')
    elif metric_name == 2 or metric_name == 6: 
        if eod_radio == 'TPR':
            heatmap_data = []
            
            for label in unique_labels:
                for group in unique_groups:
                    group_indices = np.where(groups == group)
                    group_pred = predictions[group_indices]
                    group_labels = labels[group_indices]

                    tp = np.sum((group_pred == label) & (group_labels == label))
                    fn = np.sum((group_pred != label) & (group_labels == label))
                    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

                    heatmap_data.append((str(int(label)), group, tpr))
            
            chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'TPR')
        elif eod_radio == 'FPR':
            heatmap_data = []
            
            for label in unique_labels:
                for group in unique_groups:
                    group_indices = np.where(groups == group)
                    group_pred = predictions[group_indices]
                    group_labels = labels[group_indices]

                    fp = np.sum((group_pred == label) & (group_labels != label))
                    tn = np.sum((group_pred != label) & (group_labels != label))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    heatmap_data.append((str(int(label)), group, fpr))
            
            chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'FPR')
    elif metric_name == 3 or metric_name == 7:
        group_accs = []
        for group in unique_groups:
            group_indices = np.where(groups == group)
            group_accuracy = accuracy_score(labels[group_indices], predictions[group_indices])

            group_accs.append(group_accuracy)

        data = {'Sensitive Subgroup': unique_groups, 'Accuracy': group_accs}
        chart = draw_bar_metric_view(data, 'Accuracy')

    return chart


def customize_unselected_glyph(plot, element):
    renderer = plot.state.renderers[-1]  # Get the last renderer, assuming it's your scatter plot
    # Adjusting non-selected glyph properties
    renderer.nonselection_glyph.fill_alpha = 0.
    renderer.nonselection_glyph.line_alpha = 0.
    # Adjusting selected glyph properties
    # renderer.selection_glyph.fill_alpha = 1
    # renderer.selection_glyph.line_alpha = 1 


def create_sector(center, radius, x_y_ratio, start_angle, end_angle, steps=100):
    start_angle_rad = np.radians(start_angle)
    end_angle_rad = np.radians(end_angle)
    angles = np.linspace(start_angle_rad, end_angle_rad, steps)
    points = [center]
    for angle in angles:
        x = center[0] + x_y_ratio * radius * np.cos(angle)
        y = center[1] + radius * np.sin(angle)
        points.append((x, y))
    points.append(center)
    return np.array(points)