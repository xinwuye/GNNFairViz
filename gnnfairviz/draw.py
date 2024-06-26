from . import util
from .metrics import node_classification
import numpy as np
import holoviews as hv
from holoviews import opts
import numpy as np
import pandas as pd
from bokeh.models import HoverTool
from . import RangesetCategorical
from sklearn.metrics import accuracy_score
import matplotlib.colors as mcolors


hv.extension('bokeh')
 
PERPLEXITY = 15 # german
SEED = 42

# COLOR_TRUE = '#FFEEDA' # orange
# COLOR_FALSE = '#DBE3E8' # blue
# light
# COLOR_TRUE = '#FFF7ED' # orange
# COLOR_FALSE = '#BDC5CB' # blue
# dark
COLOR_TRUE = '#DAC5A9' # orange
COLOR_FALSE = '#6E808C' # blue

# END_COLOR = '#FFDADA' # pink
# START_COLOR = '#CDE1EE' # blue
# light
# END_COLOR = '#FFDBDB' # pink
# START_COLOR = '#BDC5CB' # blue
# dark
END_COLOR = '#DAA9A9' # pink
START_COLOR = '#6E808C' # blue
# Create a custom continuous colormap
CONTINUOUS_CMAP = mcolors.LinearSegmentedColormap.from_list("custom_gradient", [START_COLOR, END_COLOR])

LINE_GREY_COLOR = '#999999'
FILL_GREY_COLOR = '#CCCCCC'  

# BAR_COLOR1 = '#FFDADA' # pink
# BAR_COLOR0 = '#D2F6D2' # green
# light
# BAR_COLOR1 = '#FFDBDB' # pink
# BAR_COLOR0 = '#D2E2D2' # green
# dark
BAR_COLOR1 = '#DAA9A9' # pink
BAR_COLOR0 = '#87AF87' # green


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
    ).opts( 
        opts.Scatter(tools=['lasso_select', 'box_select', 'tap'],  
        )
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        border=0,
        # show the toolbar at the top
        toolbar='right',
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
        toolbar='right',
        # set x and y ranges according to x and y
        xlim=(xmin - abs(xmin) * 0.1, xmax + abs(xmax) * 0.1), 
        ylim=(ymin - abs(ymin) * 0.1, ymax + abs(ymax) * 0.1), 
        xaxis=None, yaxis=None, 
        shared_axes=False,
    )

    return polys


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
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        border=0,
        # show the toolbar at the top
        toolbar='right',
        xaxis=None, yaxis=None,  
        shared_axes=False,
    )


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
    # draw the points
    points = hv.Points((x_legend, y_legend, color_column), vdims=['color']) \
        .opts(opts.Points(size=5, color='color', xaxis=None, yaxis=None)) 
    # draw the texts
    for i, text in enumerate(texts):
        points = points * hv.Text(x_legend[i] + 0.6, y_legend[i], text, halign='left', valign='center', fontsize=6)
    points.opts(
        show_legend=False, 
        toolbar=None,
        shared_axes=False,
        # x range
        xlim=(-1, 10),
        ylim=(-0.1, 0.1+1),
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        border=0
    )

    return points


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

    # Define a custom hover tool
    custom_hover = HoverTool(
        tooltips=[
            ('Value', '@{z}')  # The '@{z}' refers to the z-values in the data
        ]
    )

    heatmap = hv.HeatMap(data).opts(
        tools=[custom_hover],  
        colorbar=True, 
        cmap=CONTINUOUS_CMAP,
        xlabel=xlabel, 
        ylabel=ylabel, 
        title=title,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        shared_axes=False, 
    )
    return heatmap


def draw_density_view_scatter(graph_metrics):
    # graph_metrics is a list of tuples (num_nodes, density)
    scatter_plot = hv.Scatter(graph_metrics, kdims=['# of Nodes'], vdims=['Density'])
    return scatter_plot.opts(
                             tools=['hover', 'tap', ], 
                             active_tools=['tap'],
                             hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
                             framewise=True,
                             # set the color to #717171
                             color=FILL_GREY_COLOR,
                             size=5
                             )


def draw_dependency_view_attr_sens_bar_all(variable_data, feat_name, groups):
    # Ensure variable_data and groups are pandas Series with the same length
    if isinstance(variable_data, np.ndarray):
        variable_data = pd.Series(variable_data, name=feat_name)
    if isinstance(groups, np.ndarray): 
        groups = pd.Series(groups, name="Group")

    # Convert variable_data to string to ensure proper concatenation
    variable_data = variable_data.astype(str)
    groups = groups.astype(str)

    # Concatenate groups and variable_data into a new column
    df = pd.DataFrame({feat_name: variable_data, 'Group': groups})
    df['Sensitive Group(Attr.)'] = df['Group'] + '(' + df[feat_name] + ')'

    # Create a count column
    df['Count'] = 1

    # Group by the new concatenated column and sum the count
    aggregated_df = df.groupby('Sensitive Group(Attr.)').sum().reset_index()
    ymax = aggregated_df['Count'].max() * 1.05

    # Create the bar chart using Holoviews
    bars = hv.Bars(aggregated_df, 'Sensitive Group(Attr.)', 'Count').opts(
        xlabel='Sensitive Group(Attribute)', ylabel='Count',
        shared_axes=False,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        ylim=(0, ymax),
        invert_axes=True,
        multi_level=False,
        show_legend=False,
        fill_alpha=0,
        line_width=1.5,
        # put toolbar on the top
        toolbar='above', 
    )

    return bars


def draw_dependency_view_attr_sens_bar_selected(variable_data, feat_name, groups, selected_nodes):
    # Ensure variable_data and groups are pandas Series with the same length
    if isinstance(variable_data, np.ndarray):
        variable_data = pd.Series(variable_data, name=feat_name)
    if isinstance(groups, np.ndarray): 
        groups = pd.Series(groups, name="Group")

    # Select the specified nodes
    variable_data = variable_data.iloc[selected_nodes.astype(int)]
    groups = groups.iloc[selected_nodes.astype(int)]

    # Convert variable_data and groups to string to ensure proper concatenation
    variable_data = variable_data.astype(str)
    groups = groups.astype(str)

    # Concatenate groups and variable_data into a new column
    df = pd.DataFrame({feat_name: variable_data, 'Group': groups})
    df['Sensitive Group(Attr.)'] = df['Group'] + '(' + df[feat_name] + ')'

    # Create a count column with the same length as variable_data and all values are 1
    df['Count'] = 1

    # Group by the new concatenated column and sum the count
    aggregated_df = df.groupby('Sensitive Group(Attr.)').sum().reset_index()

    # Create the bar chart using Holoviews
    bars = hv.Bars(aggregated_df, 'Sensitive Group(Attr.)', 'Count').opts(
        xlabel='Sensitive Group(Attribute)', ylabel='Count',
        shared_axes=False,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        multi_level=False,
        # put toolbar on the top
        toolbar='above',
        line_color=None,
        fill_alpha=1,
        fill_color=FILL_GREY_COLOR,
        show_legend=False, 
        xrotation=90,
    )

    return bars


def draw_attribute_view_overview_all(feat, groups, columns_categorical, hw_ratio, selected_attrs_ls):
    n_all_nodes = len(feat)
    n_selected_attrs = len(selected_attrs_ls)
    groups = pd.Series(groups, name="Group")
    selected_nodes = np.arange(n_all_nodes)
    bias_indicators, overall_bias_indicators, ns, all_ns, all_unique_groups = util.analyze_bias(feat, groups, columns_categorical, selected_nodes)

    # ns being the number of each unique value in 
    m = len(ns)
    n = len(overall_bias_indicators)

    circle_gap = 1

    ymax = n+1.5
    ymin = 0
    xmax = n_all_nodes * 1.1 
    # solve: xmin = -circle_gap * (1 + n_selected_attrs) * ((xmax - xmin) / (ymax - ymin)) / hw_ratio
    xmin = (circle_gap * xmax * (n_selected_attrs + 1)) / (circle_gap * n_selected_attrs + circle_gap - hw_ratio * ymax + hw_ratio * ymin)
    if abs(xmin) > xmax:
        # solve: -xmax = -circle_gap * (1 + n_selected_attrs) * (2 * xmax / (ymax - ymin)) / hw_ratio
        circle_gap = (hw_ratio * (ymax - ymin)) / (2 * (1 + n_selected_attrs))
        xmin = -xmax

    # feat is a df, get the column names as feat_names
    feat_names = feat.columns

    # Prepare the data for Rectangles
    rect_data = []
    x0 = 0
    for j in range(m):
        width = ns[j]
        column = bias_indicators[:, j]
        rect_data.extend([(x0, i, x0 + width, i+1, column[i], feat_names[i]) for i in range(n)])
        x0 += all_ns[j]

    rect_data = pd.DataFrame(rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Color', 'ID'])        

    # Calculate the center of each column
    column_centers = [sum(all_ns[:i+1]) - all_ns[i]/2 for i in range(m)]
    # create yticks using column_centers and all_unique_groups
    # yticks = [(c, all_unique_groups[i]) for i, c in enumerate(column_centers)]
    custom_hover = HoverTool(tooltips=[('Attr.:', '@{ID}')])
    # Create the Rectangles plot
    plot = hv.Rectangles(rect_data, vdims=['Color', 'ID']).opts(
        opts.Rectangles(
                        # tools=['tap'], active_tools=['tap'],
                        color=hv.dim('Color').categorize(
                            {False: COLOR_FALSE, True: COLOR_TRUE}
                        ),
                        xaxis=None,  # Hide x-axis
                        line_width=0.1,
                        framewise=True,
                        tools=[custom_hover],
                        alpha=0.3
                        )).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        invert_axes=True,
        framewise=True,
        shared_axes=False,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax) 
    )
    return plot


def draw_attribute_view_overview_selected(feat, groups, columns_categorical, selected_nodes, 
                                 x, y, contributions, 
                                 contributions_selected_attrs,
                                 hw_ratio,
                                 selected_attrs_ls):
    n_all_nodes = len(feat)
    n_selected_attrs = len(selected_attrs_ls)
    groups = pd.Series(groups, name="Group")
    bias_indicators, overall_bias_indicators, ns, all_ns, all_unique_groups = util.analyze_bias(feat, groups, columns_categorical, selected_nodes)

    # ns being the number of each unique value in 
    m = len(ns)
    n = len(overall_bias_indicators)

    circle_gap = 1

    ymax = n+1.5
    ymin = 0
    xmax = n_all_nodes * 1.1 
    # solve: xmin = -circle_gap * (1 + n_selected_attrs) * ((xmax - xmin) / (ymax - ymin)) / hw_ratio
    xmin = (circle_gap * xmax * (n_selected_attrs + 1)) / (circle_gap * n_selected_attrs + circle_gap - hw_ratio * ymax + hw_ratio * ymin)
    if abs(xmin) > xmax:
        # solve: -xmax = -circle_gap * (1 + n_selected_attrs) * (2 * xmax / (ymax - ymin)) / hw_ratio
        circle_gap = (hw_ratio * (ymax - ymin)) / (2 * (1 + n_selected_attrs))
        xmin = -xmax
    # Prepare the data for Rectangles
    rect_data = []
    x0 = 0
    for j in range(m):
        width = ns[j]
        column = bias_indicators[:, j]
        rect_data.extend([(x0, i, x0 + width, i+1, column[i], i) for i in range(n)])
        x0 += all_ns[j]

    rect_data = pd.DataFrame(rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Color', 'ID'])        

    # Calculate the center of each column
    column_centers = [sum(all_ns[:i+1]) - all_ns[i]/2 for i in range(m)]
    # create yticks using column_centers and all_unique_groups
    yticks = [(c, all_unique_groups[i] + f"({int(ns[i])}/{int(all_ns[i])})") for i, c in enumerate(column_centers)]
    # Create the Rectangles plot
    plot = hv.Rectangles(rect_data, vdims=['Color', 'ID']).opts(
        opts.Rectangles(
                        color=hv.dim('Color').categorize(
                            {False: COLOR_FALSE, True: COLOR_TRUE}
                        ),
                        xaxis=None,  # Hide x-axis
                        xlabel='Sensitive Group',  # X-axis label
                        yticks=yticks,
                        line_width=0.1,
                        framewise=True,
                        ))  # Column tick labels
    
    # glyph plot
    r_sector1 = circle_gap * 0.45
    r_ellipse1 = r_sector1 * 1.5
    r_trans_circle = r_sector1 * 2
    r_selected_attr_circle = r_sector1 * 1.5
    r_selected_attr_sector = r_sector1

    # calculate the proportion of x range / y range
    x_y_ratio = (xmax - xmin) / (ymax - ymin) / hw_ratio

    circle_x = -circle_gap / 2 * x_y_ratio

    # contributions is a 1d np array, get the max absolute value
    contributions_selected_attrs = np.array(contributions_selected_attrs)
    max_contrib = np.max(np.concatenate([np.abs(contributions), np.abs(contributions_selected_attrs)]))
    # Normalize contributions to [-1, 1] 
    normalized_contributions = contributions / max_contrib
    sector_pointes1_angles = normalized_contributions * 180
    normalized_contributions_selected_attrs = contributions_selected_attrs / max_contrib
    selected_attr_sector_angles = normalized_contributions_selected_attrs * 180
 
    n = bias_indicators.shape[0]

    ellipse_data = []
    hover_data = []
    for i in range(n):
        # sector_pointes1: contributions
        end_angle = sector_pointes1_angles[i]
        sector_points1 = create_sector(center=(circle_x, i + 0.5),
                                        radius=r_sector1,
                                        x_y_ratio = x_y_ratio,
                                        start_angle=90,
                                        end_angle=90-end_angle,)
        sector_data1 = {('x', 'y'): sector_points1, 'color': LINE_GREY_COLOR}
        ellipse_data.append(sector_data1)

        # ellipse1: overall bias indicators
        # # Determine color based on row index relative to k
        color = COLOR_TRUE if overall_bias_indicators[i] else COLOR_FALSE
        e = hv.Ellipse(circle_x, i + 0.5, (x_y_ratio * r_ellipse1, r_ellipse1))
        e_data = {('x', 'y'): e.array(), 'color': color}
        ellipse_data.append(e_data)

        # transparent circle for hovering
        trans_circle = hv.Ellipse(circle_x, i + 0.5, (x_y_ratio * r_trans_circle, r_trans_circle))
        trans_circle_data = {('x', 'y'): trans_circle.array(),
                             'Bias Contribution': contributions[i],
                             'Attribute Bias': 'true' if overall_bias_indicators[i] else 'false',
                             }
        hover_data.append(trans_circle_data)

    x_data = []
    hover_data1 = []
    for i, selected_attrs in enumerate(selected_attrs_ls):
        # attr selection circles
        x_pos = circle_x - circle_gap * x_y_ratio * (i + 1)
        for selected_attr in selected_attrs:
            y_pos = selected_attr + 0.5
            # attr contribution sector
            end_angle = selected_attr_sector_angles[i]
            selected_attr_sector = create_sector(center=(x_pos, y_pos),
                                            radius=r_selected_attr_sector,
                                            x_y_ratio = x_y_ratio,
                                            start_angle=90,
                                            end_angle=90-end_angle,)
            sector_data1 = {('x', 'y'): selected_attr_sector, 'color': LINE_GREY_COLOR}
            ellipse_data.append(sector_data1)
            # inner circle
            e = hv.Ellipse(x_pos, y_pos, (x_y_ratio * r_selected_attr_circle, r_selected_attr_circle))
            e_data = {('x', 'y'): e.array(), 'color': FILL_GREY_COLOR} 
            ellipse_data.append(e_data)
            # transparent circle for hovering
            trans_circle = hv.Ellipse(x_pos, y_pos, (x_y_ratio * r_trans_circle, r_trans_circle))
            trans_circle_data = {('x', 'y'): trans_circle.array(),
                                'Bias Contribution': contributions_selected_attrs[i],
                                }
            hover_data1.append(trans_circle_data)

        # xs
        x_h_line = x_pos + circle_gap * x_y_ratio / 2
        y_pos = n + 0.5
        h_line_data = [(x_h_line, 0), (x_h_line, y_pos + 0.5)]  
        x_data.append(h_line_data)
        if selected_attrs:
            x_left = x_pos - r_sector1 * x_y_ratio
            x_right = x_pos + r_sector1 * x_y_ratio
            y_bottom = y_pos - r_sector1
            y_top = y_pos + r_sector1
            p1_data = [(x_left, y_bottom), (x_right, y_top)]
            p2_data = [(x_right, y_bottom), (x_left, y_top)]
            p_data = [p1_data, p2_data]
            x_data.extend(p_data)

    glyph_plot = hv.Polygons(ellipse_data, vdims='color').opts(
        line_width=0,
        color='color',
        framewise=True,
    )

    x_plot = hv.Path(x_data).opts(
        color='black',
        framewise=True,
    )

    trans_circles = hv.Polygons(hover_data, 
                                vdims=['Bias Contribution', 
                                       'Attribute Bias', ]).opts(
        fill_alpha=0,
        line_width=0, 
        tools=['hover'], 
        framewise=True,
    )
    trans_circles1 = hv.Polygons(hover_data1, 
                                vdims=['Bias Contribution', ]).opts(
        fill_alpha=0,
        line_width=0, 
        tools=['hover'], 
        framewise=True,
    )

    # Prepare data for transparent rectangles with black strokes
    transparent_rect_data = []
    x0 = 0  # Starting x-coordinate
    for width in all_ns:
        # Add a rectangle for each group, transparent fill and black stroke
        transparent_rect_data.append((x0, 0, x0 + width, n, LINE_GREY_COLOR))
        x0 += width

    # Convert transparent rectangle data into a DataFrame
    transparent_rect_df = pd.DataFrame(transparent_rect_data, columns=['x0', 'y0', 'x1', 'y1', 'Line_Color'])

    # Create Transparent Rectangles plot
    transparent_rectangles_plot = hv.Rectangles(transparent_rect_df, vdims=['Line_Color']).opts(
        fill_alpha=0,  # Set fill color to transparent
        line_color=hv.dim('Line_Color'),
        line_width=1, 
        tools=[], 
        framewise=True,
    ) 
    
    # Overlay Transparent Rectangles onto the existing combined plot with rectangles
    final_combined_plot = (plot * glyph_plot * x_plot * transparent_rectangles_plot * trans_circles * trans_circles1)
    
    if x:
        if 0 <= x <= n:
            # Draw an arrow pointing downwards at the top of the plot
            # Since the plot is inverted, the top is actually on the right, before inverting
            arrow_y_position = n_all_nodes*1.02  # Adjust this value as needed to place the arrow correctly
            arrow = hv.Arrow(int(x)+0.5, arrow_y_position, direction='v', arrowstyle='-|>').opts(
                framewise=True,
            )

            # Overlay the Arrow on the Final Combined Plot
            final_plot_with_arrow = final_combined_plot * arrow
        else:
            final_plot_with_arrow = final_combined_plot
    else:
        final_plot_with_arrow = final_combined_plot
    # Return the final plot with the arrow
    return final_plot_with_arrow.opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        invert_axes=True,
        framewise=True,
        shared_axes=False,
        xlim=(xmin, xmax),
        ylim=(ymin, ymax),
        tools=[] 
    )


def draw_dependency_view_attr_degree_violin(feat, computational_graph_degrees, selected_nodes, hop, scale):
    # Prepare the data
    feat_series = pd.Series(feat)
    all_data = computational_graph_degrees[hop]  # Use all data from the specified 'hop' index
    if scale == 'Log':
        all_data = np.log(all_data + 1)
    
    # Create a DataFrame for all nodes
    df_all = pd.DataFrame({
        'Attribute': feat_series,
        '# of Neighbors': all_data,
        'selected': 'Overall'  # This column specifies that the data represents overall distribution
    })
    
    # Copy selected nodes for a separate 'Selected' entry
    df_selected = df_all.iloc[selected_nodes.astype(int)].copy()
    df_selected['selected'] = 'Selected'
    
    # Concatenate the original and selected data
    df_combined = pd.concat([df_all, df_selected], ignore_index=True)
    
    # Convert 'feat' to string for categorical x-axis
    df_combined['Attribute'] = df_combined['Attribute'].astype(str)
    
    # Create the violin plot with the split condition
    violin = hv.Violin(df_combined, ['selected', 'Attribute'], '# of Neighbors')
    violin = violin.opts(opts.Violin(split=hv.dim('selected'))).opts(
        ylabel='# of Neighbors' if scale == 'Original' else 'Log(# of Neighbors)',
        show_legend=True, 
        legend_position='top',
        show_title=False,
        cmap=[BAR_COLOR1, BAR_COLOR0],
        violin_line_color=None,
        framewise=True,
        shared_axes=False,
        invert_axes=True,
        margin=(-10, 0, 0, 0),
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)] 
    )

    return violin


def draw_dependency_view_structure_sens(group_connection_matrices, hop, scale):
    group_connection_matrix = group_connection_matrices[hop]
    if scale == 'Log':
        # for each (x, y, val) in group_connection_matrix, val = log(val + 1), here group_connection_matrix is a list of tuples
        group_connection_matrix = [(x, y, np.log(val + 1)) for x, y, val in group_connection_matrix]
    custom_hover = HoverTool(
        tooltips=[
            ('# of Edges', '@{z}')  # The '@{z}' refers to the z-values in the data
        ]
    )
    # Create a HeatMap plot
    heatmap = hv.HeatMap(group_connection_matrix).opts(
        tools=[custom_hover],  
        colorbar=True, 
        # cmap='Viridis', 
        cmap=CONTINUOUS_CMAP,
        xlabel='All Nodes', 
        ylabel='Selected Nodes', 
        # rotate the x-axis labels by 90 degrees
        xrotation=90,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        shared_axes=False, 
    )
    return heatmap


def draw_dependency_view_attr_sens_violin(feat, groups, selected_nodes):
    # Prepare the data
    groups_series = pd.Series(groups)
    all_data = feat  # Use all data from the specified 'hop' index
    
    # Create a DataFrame for all nodes
    df_all = pd.DataFrame({
        'Sensitive Group': groups_series,
        'Attribute': all_data,
        'selected': 'Overall'  # This column specifies that the data represents overall distribution
    })
    
    # Copy selected nodes for a separate 'Selected' entry
    df_selected = df_all.iloc[selected_nodes.astype(int)].copy()
    df_selected['selected'] = 'Selected'
    
    # Concatenate the original and selected data
    df_combined = pd.concat([df_all, df_selected], ignore_index=True)

    # Convert 'feat' to string for categorical x-axis
    df_combined['Sensitive Group'] = df_combined['Sensitive Group'].astype(str)
    
    # Create the violin plot with the split condition
    violin = hv.Violin(df_combined, ['selected', 'Sensitive Group'], 'Attribute')
    violin = violin.opts(opts.Violin(split=hv.dim('selected'))).opts(
        show_legend=True, 
        legend_position='top',
        show_title=False,
        cmap=[BAR_COLOR1, BAR_COLOR0],
        violin_line_color=None,
        framewise=True, 
        invert_axes=True,
        shared_axes=False, 
        margin=(-10, 0, 0, 0),
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)]
    )

    return violin


def draw_dependency_view_attr_degree_hex_all(feat, computational_graph_degrees, hop, scale, gridsize=30):
    # Prepare the data
    all_data = computational_graph_degrees[hop]  # Use all data from the specified 'hop' index
    if scale == 'Log':
        all_data = np.log(all_data + 1)
    return hv.HexTiles((feat, all_data)).opts(opts.HexTiles(
        framewise=True, 
        gridsize=gridsize, 
        tools=['hover'], 
        colorbar=True)).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        shared_axes=False,
        xlabel='Attribute',
        ylabel='# of Neighbors' if scale == 'Original' else 'Log(# of Neighbors)',
        xrotation=90,
        cmap=CONTINUOUS_CMAP
    )
 

def draw_dependency_view_attr_degree_scatter_selected(feat, computational_graph_degrees, 
                                                      hop, alpha, selected_nodes, scale):
    # convert alpha (an float np array) to boolean
    alpha = alpha.astype(bool)
    x_data = feat[alpha]
    y_data = computational_graph_degrees[hop][alpha]
    if scale == 'Log':
        y_data = np.log(y_data + 1)
    if len(selected_nodes) < len(feat):
        return hv.Scatter((x_data, y_data)).opts(
            opts.Scatter(size=5, 
                        color=FILL_GREY_COLOR, 
                        line_color=None, 
                        fill_alpha=1, 
                        framewise=True,
                        shared_axes=False,
                        ylabel='# of Neighbors' if scale == 'Original' else 'Log(# of Neighbors)',
                        )
        )
    else: 
        return hv.Scatter([]).opts(
            opts.Scatter(
                        framewise=True,
                        shared_axes=False,
                        ylabel='# of Neighbors' if scale == 'Original' else 'Log(# of Neighbors)',
                        ) 
        )


def draw_fairness_metric_view_value_bar(labels, groups, selected_nodes, predictions): 
    sens = groups
    mask = selected_nodes.astype(int)
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
        hv.opts.Bars(fill_color=FILL_GREY_COLOR, stacked=False, invert_axes=True, tools=[custom_hover, 'tap']) 
    ).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        xlabel='Fairness Metric',
        ylabel='Value',
        ylim=(0, 1.05),
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
        shared_axes=False,
        show_legend=False,
        line_color=None
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
        show_legend=False,
    )

    return range_bars


def draw_fairness_metric_view_detail(metric_name, selected_nodes, groups, labels, predictions, eod_radio):
    mask = selected_nodes.astype(int)
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

        chart = draw_heatmap_metric_view(heatmap_data, 'Pred.', 'Sensitive Group', 'P(Pred.)')
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
        
        chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Group', 'TPR')
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
            
            chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Group', 'TPR')
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
            
            chart = draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Group', 'FPR')
    elif metric_name == 3 or metric_name == 7:
        group_accs = []
        for group in unique_groups:
            group_indices = np.where(groups == group)
            group_accuracy = accuracy_score(labels[group_indices], predictions[group_indices])

            group_accs.append(group_accuracy)

        data = {'Sensitive Group': unique_groups, 'Accuracy': group_accs}
        chart = draw_bar_metric_view(data, 'Accuracy')

    return chart


def draw_bar_metric_view(data, metric_name):
    # Define a custom hover tool
    custom_hover = HoverTool(
        tooltips=[
            (metric_name, '@{'+metric_name+'}')  # dynamically use the metric name for the tooltip
        ]
    )
    # Creating a bar chart
    bar_chart = hv.Bars(data, 'Sensitive Group', metric_name).opts(opts.Bars(xrotation=90)).opts(
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)], 
        fill_color=FILL_GREY_COLOR,  
        line_color=None, 
        tools=[custom_hover], 
        invert_axes=True,
        shared_axes=False, 
    )
    return bar_chart


def draw_structural_bias_overview_hist_all(frequencies_dict, edges_dict, hop, scale):
    frequencies = frequencies_dict[scale][hop]
    edges = edges_dict[scale][hop]
    hist = hv.Histogram((edges, frequencies)).opts(
        xlabel='# of Neighbors' if scale == 'Original' else 'Log(# of Neighbors)',
        ylabel='Frequency',
        title='',
        framewise=True,
        shared_axes=False,
        tools=['xbox_select'],
        fill_alpha=0,
        line_width=1.5,
        line_alpha=1,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
    )
    return hist


def draw_structural_bias_overview_hist_selected(computational_graph_degrees, edges_dict, hop, 
                                                selected_nodes,
                                                scale):
    if len(selected_nodes) == 0:
        data = computational_graph_degrees[hop]
    else:
        data = computational_graph_degrees[hop][selected_nodes.astype(int)]
    if scale == 'Log':
        data = np.log(data + 1)
    edges = edges_dict[scale][hop]
    frequencies, edges = np.histogram(data, bins=edges)
    hist = hv.Histogram((edges, frequencies)).opts(
        xlabel='# of Neighbors' if scale == 'Original' else 'Log(# of Neighbors)',
        ylabel='Frequency',
        title='',
        color=FILL_GREY_COLOR,
        framewise=True,
        shared_axes=False,
        fill_alpha=1,
        line_width=0,
        hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
    )
    return hist


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

