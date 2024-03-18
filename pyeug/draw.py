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
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
from scipy import stats
from sklearn.metrics import f1_score, accuracy_score
import matplotlib.pyplot as plt
from bokeh.models import HoverTool
import matplotlib.colors as mcolors

hv.extension('bokeh')

PERPLEXITY = 15 # german
SEED = 42

# def draw_graph_view(xy, source_id, target_id, layer, sens, 
#                     sens_names, sens_name, bundle, sampled_alpha, colors):
#     node_size = 3
#     # print(layer)
#     # print(xy)
#     current_xy = xy[layer]
#     x = current_xy[:,0]
#     y = current_xy[:,1]
#     node_indices = np.arange(0, len(x))
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

#     # sens_name_idx = sens_names.index(sens_name)
#     # sens_selected = sens[sens_name_idx]
#     sens_selected_unique = np.unique(sens_selected)
#     print(sens_selected_unique)
#     colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
#     nodes = hv.Nodes((x, y, node_indices, sens_selected, sampled_alpha), vdims=['Sens', 'sampled_alpha']) \
#         .opts(opts.Nodes(alpha='sampled_alpha', nonselection_alpha=0))
#     graph = hv.Graph(((source_id, target_id), nodes, ))
#     graph.nodes.opts(opts.Nodes(color='Sens', alpha='sampled_alpha', cmap=colormap, size=node_size, line_color=None,))
#     _ = graph.opts(cmap=colormap, 
#                    node_size=node_size, 
#                    edge_line_width=0.1,
#                 # node_line_color='gray', 
#                 node_color='Sens', 
#                 node_alpha='sampled_alpha')
#     if bundle:
#         g = bundle_graph(graph)
#     else:
#         g = graph

#     # draw poly
#     poly_df = RangesetCategorical.compute_contours(colormap, current_xy, sens_selected, threshold=10)
#     polys = hv.Polygons([{('x', 'y'): list(zip(poly_df.iloc[i]['xs'][0][0], poly_df.iloc[i]['ys'][0][0])), 'level': poly_df.iloc[i]['color']} for i in range(len(poly_df))], vdims='level')
#     polys.opts(color='level', line_width=0, alpha=0.5)

#     ret = (g * polys * g.nodes).opts(
#         opts.Nodes(color='Sens', alpha='sampled_alpha', size=node_size, height=300, width=300, cmap=colors, 
#             # legend_position='right', 
#             show_legend=False,
#             tools=['lasso_select', 'box_select']),
#         opts.Graph(edge_line_width=0.1, inspection_policy='nodes', 
#             edge_hover_line_color='green', node_hover_fill_color='red')
#         ).opts(xlabel='x_graph_view', ylabel='y_graph_view', border=0,
#                hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)])
#     # ret = (g * polys).opts(
#     #     opts.Nodes(color='Sens', alpha='sampled_alpha', size=1, height=300, width=300, cmap=colors, 
#     #         # legend_position='right', 
#     #         show_legend=False,
#     #         tools=['lasso_select', 'box_select']),
#     #     opts.Graph(edge_line_width=0.1, inspection_policy='nodes', 
#     #         edge_hover_line_color='green', node_hover_fill_color='red')
#     #     ).opts(xlabel='x_graph_view', ylabel='y_graph_view', border=0)

#     return ret


def draw_embedding_view(xy, layer, groups, sampled_alpha, colors):
    node_size = 10
    current_xy = xy[layer]
    x = current_xy[:, 0]
    y = current_xy[:, 1]

    sens_selected = groups
    sens_selected_unique = np.unique(sens_selected)

    colormap = {sens_selected_unique[i]: colors[i] for i in range(len(sens_selected_unique))}
    color_column = [colormap[group] for group in groups]

    # Prepare DataFrame with color information
    scatter_data = {'x': x, 'y': y, 'color': color_column, 'alpha': sampled_alpha}
    df = pd.DataFrame(scatter_data)
    scatter = hv.Scatter(df, kdims=['x', 'y'], vdims=['color', 'alpha']).opts(
        size=node_size, 
        color='color', 
        alpha='alpha', 
        line_color=None)

    # Calculate and draw polygons (regions or clusters) as before
    poly_df = RangesetCategorical.compute_contours(colormap, current_xy, sens_selected, threshold=10)
    polys = hv.Polygons([{('x', 'y'): list(zip(poly_df.iloc[i]['xs'][0][0], poly_df.iloc[i]['ys'][0][0])), 
                          'level': poly_df.iloc[i]['color']} for i in range(len(poly_df))], vdims='level')
    polys.opts(color='level', line_width=0, alpha=0.5)

    # Combine scatterplot and polygons
    plot = (scatter * polys).opts(
        opts.Scatter(tools=['lasso_select', 'box_select'], 
                    #  show_legend=True,
                    #  legend_position='right',
                     )
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        show_legend=True,
        border=0,
        # show the toolbar at the top
        toolbar='above',
    )
    # plot = scatter

    return plot


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
    print(texts)
    print(color_column)
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
    bar_chart = hv.Bars(data, 'Sensitive Subgroup', metric_name).opts(opts.Bars(xrotation=90))
    return bar_chart.opts(hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)])


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


def draw_correlation_view_hex(index, feat, metric, selected_nodes, gridsize=30):
    x_data = feat[index][selected_nodes]
    y_data = metric[selected_nodes]
    return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(gridsize=gridsize, tools=['hover'], colorbar=True)).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
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
    scatter_plot = hv.Scatter(graph_metrics, kdims=['Number of Nodes'], vdims=['Density'])
    return scatter_plot.opts(
                             tools=['hover', 'tap', ], 
                             active_tools=['tap'],
                             hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                             framewise=True,
                             # set the color to #717171
                                color='#717171',
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
    if len(selected_nodes) != adj.shape[0]:
        # Extract non-zero indices
        rows, cols = adj[selected_nodes][: , selected_nodes].nonzero()

        # Prepare data for Holoviews
        # We create a list of tuples (i, j, 1) for each non-zero element
        data = [(rows[i], cols[i], 1) for i in range(len(rows))]

        # Define a Holoviews Dataset
        hv_dataset = hv.Dataset(data, ['i', 'j'], 'value')

        # Create a heatmap using the dataset
        heatmap = hv.HeatMap(hv_dataset).opts(
            colorbar=False,
            # set the color to #717171
            color='#717171',
        )
    else:
        heatmap = hv.HeatMap([])

    return heatmap.opts(hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                        shared_axes=False,
                        xaxis=None,
                        yaxis=None,
                        framewise=True,
                        border=0,).redim.range(x=(0, len(selected_nodes)), y=(0, len(selected_nodes)))


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
    print(df)
    
    # Draw the violin plot for the specified feature, grouped by 'Group'
    violin = hv.Violin(df, 'Group', feat_name).opts(ylabel=feat_name,
                                                    hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                                                    shared_axes=False,
                                                    xrotation=90,)
    
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


def draw_attribute_view_bar(variable_data, feat_name, groups, selected_nodes):
    # Ensure variable_data and groups are pandas Series with the same length
    if isinstance(variable_data, np.ndarray):
        variable_data = pd.Series(variable_data, name=feat_name)
    if isinstance(groups, np.ndarray):
        groups = pd.Series(groups, name="Group")

    variable_data = variable_data.iloc[selected_nodes]
    groups = groups.iloc[selected_nodes]

    # Step 1: Construct a Contingency Table
    contingency_table = pd.crosstab(variable_data, groups)

    # Step 2: Perform the Chi-Square Test
    chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)

    # Pairwise tests (Chi-Square) for each group
    group_colors = {}
    for group_name in groups.unique():
        group_data = contingency_table[group_name]
        other_data = contingency_table.drop(columns=group_name)
        stat, p, dof, expected = chi2_contingency(pd.concat([group_data, other_data], axis=1))
        group_colors[group_name] = 'green' if p > 0.05 else 'red'
    
    # Combine into a DataFrame
    df = pd.concat([variable_data, groups], axis=1)
    
    # Count occurrences of each category within each group
    aggregated_df = df.groupby(['Group', feat_name]).size().reset_index(name='Count')
    
    # Create the bar chart
    bars = hv.Bars(aggregated_df, ['Group', feat_name], 'Count').opts(stacked=False, width=400, tools=['hover'],
                                                                      xlabel='Group', ylabel='Count',
                                                                      title=f'{feat_name} (p={chi2_p:.3f})',
                                                                      shared_axes=False,
                                                                      color=hv.dim('Group').categorize(group_colors),
                                                                      hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                                                                      xrotation=90)
    if chi2_p <= 0.05:
        bars.opts(hooks=[ylabel_red_hook])
    else:
        bars.opts(hooks=[ylabel_black_hook])
    return bars


# def draw_attribute_view_sens(groups, selected_nodes):
#     """
#     Draws a bar chart for the categorical values contained in a NumPy array using HoloViews.

#     Parameters:
#     - groups: np.array, an array consisting of categorical values.

#     Returns:
#     A HoloViews object that renders a bar chart visualizing the frequency of each category.
#     """
#     groups = groups[selected_nodes]
#     # Calculate the frequency of each unique category in the array
#     unique, counts = np.unique(groups, return_counts=True)
    
#     # Prepare the data for HoloViews in the form of a list of tuples (category, frequency)
#     data = list(zip(unique, counts))
    
#     # Create a Bar chart using HoloViews
#     bars = hv.Bars(data, hv.Dimension('Category'), 'Frequency').opts(
#         tools=['hover'],  # Enable hover tool for more interactive charts
#         xlabel='Category', ylabel='Frequency',  # Label axes
#         title='Categorical Value Frequencies',  # Chart title
#         shared_axes=False, width=400
#     )
    
#     return bars

def draw_attribute_view_overview(feat, groups, columns_categorical, selected_nodes, individual_bias_metrics, x, y):
    # print(groups)
    groups = pd.Series(groups, name="Group")
    bias_indicators, overall_bias_indicators, ns = util.analyze_bias(feat, groups, columns_categorical, selected_nodes)
    # # k being the sum of overall_bias_indicator
    # k = len(overall_bias_indicators) - overall_bias_indicators.sum()
    # ns being the number of each unique value in 
    m = len(ns)
    # print(ns)
    n = len(overall_bias_indicators)
    # Prepare the data for Rectangles
    rect_data = []
    x0 = 0
    for j in range(m):
        # print(x0)
        width = ns[j]
        column = bias_indicators[:, j]
        # print([(x0, i, x0 + width, i+1, column[i]) for i in range(n)][-4])
        rect_data.extend([(x0, i, x0 + width, i+1, column[i], i) for i in range(n)])
        x0 += width

    rect_data = pd.DataFrame(rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Color', 'ID'])        

    # Calculate the center of each column
    column_centers = [sum(ns[:i+1]) - ns[i]/2 for i in range(m)]

    # Create the Rectangles plot
    plot = hv.Rectangles(rect_data, vdims=['Color', 'ID']).opts(
        opts.Rectangles(tools=['tap'], active_tools=['tap'],
                        color=hv.dim('Color').categorize(
                            {False: 'green', True: 'red'}),
                        # yformatter='%.0f',  # Show integers on y-axis
                        # yticks=list(range(n)),  # Set y-axis ticks
                        # yaxis=None,  # Hide y-axis
                        xaxis=None,  # Hide x-axis
                        xlabel='Groups',  # X-axis label
                        yticks=[(c, str(chr(97+i))) for i, c in enumerate(column_centers)],
                        line_width=0.1,
                        # xrotation=90
                        ))  # Column tick labels

    # Add tap tool callback to the plot
    plot.opts(opts.Rectangles(
        active_tools=['tap'],
        tools=['tap'],))

   # Step 1: Correctly Generate Circle Data with Accurate Color Assignment
    # Calculate the x-coordinate for circle centers (to the left of the first column)
    # circle_x = -ns[0] / 2
    circle_x = -len(selected_nodes) / 20 - 2
    n = bias_indicators.shape[0]
    circle_data_corrected = []
    for i in range(n):
        # # Determine color based on row index relative to k
        # color = 'green' if i < k else 'red'
        color = 'red' if overall_bias_indicators[i] else 'green'
        # Calculate y-coordinate for the center of the circle (midpoint of the rectangle's height)
        y_center = i + 0.5
        # Append circle information (x-coordinate, y-coordinate, and color)
        circle_data_corrected.append((circle_x, y_center, color, i))

    circle_data_corrected = pd.DataFrame(circle_data_corrected, columns=['x', 'y', 'Color', 'ID'])

    # Step 2: Create Corrected Circles Plot
    circles_plot_corrected = hv.Scatter(circle_data_corrected, ['x', 'y'], ['Color', 'ID']).opts(
        color='Color', marker='circle', size=1,
        tools=[],  # No interactive tools needed for circles
        legend_position='top_right')

    # Step 3: Overlay Corrected Circles Plot onto the Existing Rectangles Plot
    combined_plot_corrected = plot * circles_plot_corrected

    # return combined_plot_corrected.opts(hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)])

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
        correlations.append(correlation)
        p_values.append(p_val)

    # Calculate the x-coordinate for the rectangles (to the left of the circles)
    square_x = -len(selected_nodes) / 10 - 5  # This will need to be adjusted to prevent overlap with circles

    # Prepare Rectangle Data to mimic Squares
    rectangle_data = []
    cmap = plt.get_cmap('coolwarm')
    for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
        # Convert the RGBA color from the colormap to a hexadecimal color
        fill_color = mcolors.to_hex(cmap((corr + 1) / 2)) if not np.isnan(corr) else 'white'
        line_color = 'red' if p_val < 0.05 else 'black'  # No line color if p-value is not significant

        # Append rectangle data
        rectangle_data.append((square_x, i, square_x + 10, i + 1, fill_color, line_color, corr, i))

    # After the loop, convert the list of tuples to a DataFrame
    rectangle_df = pd.DataFrame(rectangle_data, columns=['x0', 'y0', 'x1', 'y1', 'Fill_Color', 'Line_Color', 'Correlation', 'ID'])

    # Create the Rectangles plot to mimic Squares
    rectangles_plot = hv.Rectangles(rectangle_df, vdims=['Fill_Color', 'Line_Color', 'ID']).opts(
        fill_color=hv.dim('Fill_Color'),  # Fill rectangles with the specified color
        line_color=hv.dim('Line_Color'),  # Add a border around the rectangles
        line_width=0.5,
        # fill_alpha=0.5  # Adjust transparency as needed
    )
    
    # Overlay Rectangles Plot onto the existing combined plot with circles
    combined_plot_with_rectangles = (combined_plot_corrected * rectangles_plot).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        invert_axes=True,
    )
    # print(combined_plot_with_rectangles.data)
    
    # Return the updated plot with rectangles as squares
    return combined_plot_with_rectangles


# def draw_attribute_view_correlation_violin(feat, metric, selected_nodes):
#     x_data = feat[selected_nodes]
#     y_data = metric[selected_nodes]
#     data = pd.DataFrame({'feat': x_data, 'metric': y_data})
#     violin_list = [hv.Violin(data[data['feat'] == i], vdims='metric') for i in data['feat'].unique()]
#     violin_plot = hv.Overlay(violin_list)
#     violin_plot.opts(opts.Violin(tools=['hover'], cmap='Category20')).opts(
#         hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
#     )

#     return violin_plot


def draw_attribute_view_correlation_violin(feat, metric, selected_nodes):
    x_data = feat[selected_nodes]
    y_data = metric[selected_nodes]
    data = pd.DataFrame({'feat': x_data, 'metric': y_data})
    violin_plot = hv.Violin(data, 'feat', 'metric').opts(
        xlabel='Feature', ylabel='Metric').opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
    )

    return violin_plot


def draw_attribute_view_correlation_hex(feat, metric, selected_nodes, gridsize=30):
    x_data = feat[selected_nodes]
    y_data = metric[selected_nodes]
    return hv.HexTiles((x_data, y_data)).opts(opts.HexTiles(
        gridsize=gridsize, 
        tools=['hover'], 
        colorbar=True)).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        shared_axes=False,
    )


# def draw_fairness_metric_view_bar(labels, groups, selected_nodes, predictions):
#     sens = groups

#     # mask = self._get_data_mask()
#     mask = selected_nodes
#     predictions = predictions[mask]
#     labels = labels[mask]
#     sens = sens[mask]
#     labeled_mask = labels != -1
#     labeled_predictions = predictions[labeled_mask]
#     labeled_labels = labels[labeled_mask]
#     labeled_sens = sens[labeled_mask]

#     letter_values = {'SD MF1': [node_classification.micro_f1_std_dev(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
#                     #  'DI': node_classification.surrogate_di(labeled_predictions, labeled_labels, labeled_sens),
#                         'EFPR': [node_classification.efpr(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
#                         'EFNR': [node_classification.efnr(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
#                     #  'ETPR': node_classification.etpr(labeled_predictions, labeled_labels, labeled_sens),
#                         'D SP': [node_classification.delta_dp_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
#                         'Dm SP': [node_classification.max_diff_delta_dp(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#                         'D EO': [node_classification.delta_eo_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
#                         'D Acc': [node_classification.delta_accuracy_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
#                         'SD Acc': [node_classification.sigma_accuracy_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
#                     #  'D TED': node_classification.delta_ted_metric(labeled_predictions, labeled_labels, labeled_sens),
#                         }
#     # Prepare data
#     data = {
#         'Metric': [],
#         'Value': [],
#         'Lower Bound': [],
#         'Upper Bound': []
#     }
#     for letter, [value, bounds] in letter_values.items():
#         data['Metric'].append(letter)
#         data['Value'].append(value)
#         data['Lower Bound'].append(bounds[0])
#         data['Upper Bound'].append(bounds[1])
    
#     df = pd.DataFrame(data)
    
#     # Create the transparent bars (range)
#     range_bars = hv.Bars(df, kdims=['Metric'], vdims=['Upper Bound']).opts(
#         hv.opts.Bars(fill_alpha=0, line_dash='dashed', line_width=1.5)
#     )
    
#     # Create the value bars
#     value_bars = hv.Bars(df, kdims=['Metric'], vdims=['Value']).opts(
#         hv.opts.Bars(fill_color='blue')
#     )
    
#     # Overlay the bars to create the composite chart
#     composite_chart = (range_bars * value_bars)
    
#     return composite_chart


def draw_fairness_metric_view_bar(x, y, labels, groups, selected_nodes, predictions):
    sens = groups

    # mask = self._get_data_mask()
    mask = selected_nodes
    predictions = predictions[mask]
    labels = labels[mask]
    sens = sens[mask]
    labeled_mask = labels != -1
    labeled_predictions = predictions[labeled_mask]
    labeled_labels = labels[labeled_mask]
    labeled_sens = sens[labeled_mask]

    letter_values = {
                        # 'SD MF1': [node_classification.micro_f1_std_dev(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                    #  'DI': node_classification.surrogate_di(labeled_predictions, labeled_labels, labeled_sens),
                        # 'EFPR': [node_classification.efpr(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                        # 'EFNR': [node_classification.efnr(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                    #  'ETPR': node_classification.etpr(labeled_predictions, labeled_labels, labeled_sens),
                        'D SP': [node_classification.delta_dp_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                        'Dm SP': [node_classification.max_diff_delta_dp(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
                        'D EO': [node_classification.delta_eo_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                        'D Acc': [node_classification.delta_accuracy_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
                        'SD Acc': [node_classification.sigma_accuracy_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                    #  'D TED': node_classification.delta_ted_metric(labeled_predictions, labeled_labels, labeled_sens),
                        }

    data = {
        'Metric': [],
        'Percentage Value': [],  # Holds the percentage values
        'Value': [],  # Actual metric values
        'Upper Bound Percentage': []  # Actual upper bounds
    }
    label_data = {
        'x': [],
        'y': [],
        'text': []
    }

    for letter, [value, bounds] in letter_values.items():
        percentage_value = (value / bounds[1]) * 100  # Calculate percentage
        data['Metric'].append(letter)
        data['Percentage Value'].append(percentage_value)
        data['Value'].append(value)
        data['Upper Bound Percentage'].append(100)

        # Calculate position for labels (slightly above the bars)
        value_position = percentage_value + 8  # Adjust as needed for visibility
        upper_bound_position = 106  # Slightly above 100%

        # Append data for labels showing actual values and upper bounds
        label_data['x'].extend([letter, letter])
        label_data['y'].extend([value_position, upper_bound_position])
        label_data['text'].extend([f'{value:.2f}', f'{bounds[1]}'])
    
    df = pd.DataFrame(data)
    labels_df = pd.DataFrame(label_data)

    # Create the bars
    range_bars = hv.Bars(df, kdims=['Metric'], vdims=['Upper Bound Percentage']).opts(
        hv.opts.Bars(fill_alpha=0, line_dash='dashed', line_width=1.5, invert_axes=True)
    )
    value_bars = hv.Bars(df, kdims=['Metric'], vdims=['Percentage Value']).opts(
        hv.opts.Bars(fill_color='#717171', invert_axes=True)
    )

    # Create labels
    labels = hv.Labels(labels_df, kdims=['x', 'y'], vdims=['text']).opts(
        hv.opts.Labels(text_font_size='8pt', text_align='center', text_baseline='bottom')
    )

    # Combine the bars and labels
    composite_chart = (range_bars * value_bars * labels).opts(
        hv.opts.Overlay(show_legend=False)
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
    )

    return composite_chart


def draw_fairness_metric_view_detail(metric_name_dict, metric_name, selected_nodes, groups, labels, predictions):
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
    # get the name of the clicked button
    # metric_name = event.obj.name
    # if metric_name_dict[metric_name] == 'micro_f1_std_dev':
    #     micro_f1_scores = []
    #     for group in unique_groups:
    #         group_indices = np.where(groups == group)
    #         group_pred = predictions[group_indices]
    #         group_labels = labels[group_indices]

    #         micro_f1 = f1_score(group_labels, group_pred, average='micro')
    #         micro_f1_scores.append(micro_f1)
    #     data = {'Sensitive Subgroup': unique_groups, 'Micro-F1': micro_f1_scores}
    #     chart = draw_bar_metric_view(data, 'Micro-F1')
    # elif metric_name_dict[metric_name] == 'efpr':
    #     heatmap_data = []
        
    #     for label in unique_labels:
    #         for group in unique_groups:
    #             group_indices = np.where(groups == group)
    #             group_pred = predictions[group_indices]
    #             group_labels = labels[group_indices]

    #             tn = np.sum((group_pred != label) & (group_labels != label))
    #             fp = np.sum((group_pred == label) & (group_labels != label))
    #             fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

    #             heatmap_data.append((str(int(label)), group, fpr))
        
    #     chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'FPR')
    # elif metric_name_dict[metric_name] == 'efnr':
    #     heatmap_data = []
        
    #     for label in unique_labels:
    #         for group in unique_groups:
    #             group_indices = np.where(groups == group)
    #             group_pred = predictions[group_indices]
    #             group_labels = labels[group_indices]

    #             fn = np.sum((group_pred != label) & (group_labels == label))
    #             tp = np.sum((group_pred == label) & (group_labels == label))
    #             fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    #             heatmap_data.append((str(int(label)), group, fnr))
        
    #     chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'FNR')
    if metric_name_dict[metric_name] == 'delta_dp_metric' or metric_name_dict[metric_name] == 'max_diff_delta_dp':
        heatmap_data = []

        for label in unique_labels:
            for group in unique_groups:
                group_indices = np.where(groups == group)
                group_predictions = predictions[group_indices]

                prob = np.mean(group_predictions == label)
                heatmap_data.append((str(int(label)), group, prob))

        chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'P(Pred.=Label)')
    elif metric_name_dict[metric_name] == 'delta_eo_metric':
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
        
        chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'ETPR')
    elif metric_name_dict[metric_name] == 'delta_accuracy_metric' or metric_name_dict[metric_name] == 'sigma_accuracy_metric':
        group_accs = []
        for group in unique_groups:
            group_indices = np.where(groups == group)
            group_accuracy = accuracy_score(labels[group_indices], predictions[group_indices])

            group_accs.append(group_accuracy)

        data = {'Sensitive Subgroup': unique_groups, 'Accuracy': group_accs}
        chart = draw_bar_metric_view(data, 'Accuracy')
    elif metric_name_dict[metric_name] == 'delta_ted_metric':
        heatmap_data = []

        for label in unique_labels:
            for group in unique_groups:
                group_indices = np.where(groups == group)
                fp = np.sum((predictions[group_indices] == label) & (labels[group_indices] != label))
                fn = np.sum((predictions[group_indices] != label) & (labels[group_indices] == label))

                ratio = fp / fn if fn > 0 else float('inf')
                heatmap_data.append((str(int(label)), group, ratio))

        chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'FP/FN Ratio')

    return chart