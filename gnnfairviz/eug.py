from . import util
from . import draw
from . import community
from .css import scrollbar_css, multichoice_css, switch_css, card_css, tabs_css
from . import RangesetCategorical
from . import individual_bias
import numpy as np
import holoviews as hv
import torch
import dgl
import numpy as np
from holoviews.streams import Stream, param
from holoviews import streams
import pandas as pd
import panel as pn
import sys

sys.setrecursionlimit(1000000)

hv.extension('bokeh')
pn.extension()

pn.config.raw_css.append(scrollbar_css)
pn.config.raw_css.append(multichoice_css)
pn.config.raw_css.append(switch_css)
pn.config.raw_css.append(card_css)
pn.config.raw_css.append(tabs_css)

PERPLEXITY = 15 # german
SEED = 42

FILL_GREY_COLOR = '#CCCCCC'  

# set np seed
np.random.seed(SEED)


class DataDensityView(Stream):
    graph_metrics = param.List(default=[], constant=True, doc='Graph metrics.')


class DataEmbeddingViewAlpha(Stream):
    alpha = param.Array(default=np.array([]), constant=True, doc='Sampled alpha.')


class DataEmbeddingViewPolys(Stream):
    polygons_lst = param.List(default=[], constant=True, doc='Polygons.')
    max_edges_lst = param.List(default=[], constant=True, doc='Max edges.')


class DataEmbeddingViewXy(Stream):
    xy = param.Array(default=np.array([]), constant=True, doc='XY positions.')


class DataEmbeddingViewThrRange(Stream):
    min_thr = param.Number(default=0, constant=True, doc='Min threshold.')
    max_thr = param.Number(default=1, constant=True, doc='Max threshold.')


class SelectedNodes(Stream):
    selected_nodes = param.Array(default=np.array([], dtype=int), constant=False, doc='Selected nodes.')


class Groups(Stream):
    groups = param.Array(default=np.array([]), constant=False, doc='Groups.')


class Contributions(Stream):
    contributions = param.Array(default=np.array([]), constant=False, doc='Contributions.')
    contributions_selected_attrs = param.Array(default=np.array([0.]), constant=False, doc='Contributions selected attributes.')


class SelectedAttrsLs(Stream):
    selected_attrs_ls = param.List(default=[[]], constant=False, doc='Selected attributes list.')


class GroupConnectionMatrices(Stream):
    group_connection_matrices = param.List(default=[], constant=False, doc='Group connection matrices.')


class EUG:
    def __init__(self, model, adj, feat, sens, sens_names, masks, 
                 labels, emb_fnc, perplexity=PERPLEXITY, feat_names=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if feat_names is None
        if feat_names is None:
            self.feat_names = np.array([str(i) for i in range(feat.shape[-1])])
        else:
            self.feat_names = feat_names    
        self.model = model.to(device)
        self.model.eval()
        self.adj = adj.to(device)
        self.adj_mul_indices = []
        self.adj0, self.adj1, self.adj0_scipy, self.adj1_scipy = util.modify_sparse_tensor_scipy(adj)
        self.adj0 = self.adj0.coalesce()
        self.feat = feat.to(device)
        self.n_feat = feat.shape[-1]
        self.sens = sens
        self.groups_stream = Groups(groups=self.sens[0])
        self.sens_names = sens_names
        self.train_mask = masks[0]
        self.val_mask = masks[1] 
        self.test_mask = masks[2]
        self.other_mask = ~(self.train_mask + self.val_mask + self.test_mask)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # check if adj is pytorch sparse tensor type
        if adj.is_sparse:
            src, dst = adj.coalesce().indices()
            # Create the heterograph
            g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
            g = g.int().to(device)

        self.g = g
        with torch.no_grad():
            logits = self.model(self.g, self.feat)
        self.logits = logits
        self.predictions = torch.argmax(logits, dim=1).cpu().numpy()

        if isinstance(labels, torch.Tensor):
            self.labels = labels.cpu().numpy()
        else:
            self.labels = labels

        self.neighbors = None

        preprocessed = util.init(model, feat, g, emb_fnc)
        self.embeddings = preprocessed['embeddings']
        self.max_hop = len(self.embeddings)

        self.embeddings_pca = util.proj_emb(self.embeddings, 'pca')
        self.embeddings_tsne = []
        self.embeddings_umap = util.proj_emb(self.embeddings, 'umap')

        self.xy = np.array(self.embeddings_umap)
        self.layers = list(range(1, len(self.embeddings)+1))

        self.colors = ['#F1B8B8', '#81A995', '#F1E9B8', '#907FA3', '#F1DAB8', 
                       '#7F89A1', '#D6E4AE', '#CB9AB0', '#738E96', '#F1E2B8', 
                       '#9A799E', '#B5D4A1', '#F1CEB8', '#8984A7', '#F1F1B8', ]
        
        self.n_nodes = adj.shape[0]
        self.node_indices = np.arange(self.n_nodes)
        self.selected_nodes_stream = SelectedNodes(selected_nodes=self.node_indices)
        self.selected_nodes_stream.add_subscriber(self._selected_nodes_stream_subscriber)
        self.pre_selected_nodes_stream = SelectedNodes()
        self.pre_selected_nodes_stream.add_subscriber(self._pre_selected_nodes_stream_subscriber)

        self.previous_attribute_view_overview_tap_x = None
        self.previous_attribute_view_overview_tap_y = None

        self.graph_view_scatter = None
        self.graph_view = None
        self.metric_view = None
        self.correlation_view_selection_continuous = None

        self.records = []

        self.pre_compute_contours_cache = {}

        options_dict = {f'Hop-{i+1}': i for i in range(self.max_hop)}
        self.correlation_view_selection = pn.widgets.Select(name='Hops', options=options_dict,)

    def show(self):
        height = 900
        width = 1430
        self.height = height
        self.width = width
        padding = 15
        self.line_space = 10
        self.control_panel_height = int(height/3-35)-padding
        self.control_panel_width = int(width/4-20)
        self.graph_view_height = int(height/3-35)-padding
        self.graph_view_width = int(width/4)-padding
        self.fairness_metric_view_height = int(height*2/3-35)-padding
        self.fairness_metric_view_width = int(width/4)-padding
        self.node_selection_view_height = int(height/3-35)-padding
        self.node_selection_view_width = int(width*3/4)-padding
        self.density_view_height = int(height/3-35)-padding
        self.density_view_width = int(width/4)-padding
        self.structural_bias_overview_height = int(height*2/3-35)-padding
        self.structural_bias_overview_width = int(width/4)-padding
        self.correlation_view_height = int(height/3-35)-padding
        self.correlation_view_width = int(width/4)-padding
        self.diagnostic_panel_height = int(height*2/3-35)-padding
        self.diagnostic_panel_width = int(width*3/4)-padding

        # widget for all
        self.record_button = pn.widgets.Button(name='Record', button_type='default')
        self.record_button.on_click(self._record_button_callback)
        self.sens_name_selector = pn.widgets.MultiChoice(name='Sensitive Attribute', 
                                                         options=self.sens_names, 
                                                         value = [self.sens_names[0], ], 
                                                         width=int(self.control_panel_width*0.63))
        self.previous_sens_name_value = self.sens_name_selector.value
        sens_name_confirm_control_panel_button = pn.widgets.Button(name='Confirm', button_type='default')
        sens_name_confirm_control_panel_button.on_click(self._sens_name_confirm_control_panel_button_callback)
        node_selection_confirm_control_panel_button = pn.widgets.Button(name='Confirm', button_type='default')
        node_selection_confirm_control_panel_button.on_click(self._node_selection_confirm_control_panel_button_callback)
        node_selection_clear_control_panel_button = pn.widgets.Button(name='Clear', button_type='default')
        node_selection_clear_control_panel_button.on_click(self._node_selection_clear_control_panel_button_callback)
        self.n_neighbors_scale_group = pn.widgets.RadioButtonGroup(options=['Original', 'Log'], value='Original', button_type='default')

        # widget for fairness metric view
        # a multi-select widget for selecting "train", "val", "test", "other"
        self.data_selector = pn.widgets.MultiChoice(name='Data', value=['Train', 'Val', 'Test', 'Unlabeled'],
            options=['Train', 'Val', 'Test', 'Unlabeled'],
            width=int(self.control_panel_width*0.63),
            )
        self.previous_data_selector_value = self.data_selector.value

        # latex for number of selected nodes
        self.n_selected_nodes_latex = pn.pane.LaTeX(f'# of Selected Nodes: {int(0)}/{self.n_nodes} (100%)')

### graph view
        # widget for graph view
        self.layer_selection = pn.widgets.Select(options=self.layers, name='Hop', value=self.layers[-1], width=200)

        self.projection_selection = pn.widgets.Select(options=['UMAP', 'PCA', 't-SNE'], name='Projection', width=200)
        # watch projection selection
        self.projection_selection.param.watch(self._projection_selection_callback, 'value')

        self.node_sample_size_slider = pn.widgets.IntSlider(name='Node Sample Size', 
                                                            start=0, 
                                                            end=self.n_nodes, 
                                                            step=1, 
                                                            value=300,
                                                            width=200)
        
        # create sampled_alpha
        node_sample_size = self.node_sample_size_slider.value
        self.sampled_alpha = np.zeros(self.n_nodes)
        sample_idx = np.random.choice(np.arange(self.n_nodes), node_sample_size, replace=False)
        self.sampled_alpha[sample_idx] = 1
        self.data_embedding_view_alpha = DataEmbeddingViewAlpha(alpha=self.sampled_alpha) 

        sample_button = pn.widgets.Button(name='Sample', button_type='default')
        sample_button.on_click(self._sample_button_callback)

        self.data_embedding_view_xy = DataEmbeddingViewXy(xy=self.xy)
        self.data_embedding_view_xy.add_subscriber(self._prepare_data_embedding_view)
        self.groups_stream.add_subscriber(self._prepare_data_embedding_view)
        self.layer_selection.param.watch(self._prepare_data_embedding_view, 'value') 
        self.data_embedding_view_polys = DataEmbeddingViewPolys()

        self.data_embedding_view_thr_range = DataEmbeddingViewThrRange()
        self.data_embedding_view_thr_range.add_subscriber(self._update_thr_range)

        self.embedding_view_thr_slider = pn.widgets.FloatSlider(name='Rangeset Threshold', 
                                                                start=self.data_embedding_view_thr_range.min_thr, 
                                                                end=self.data_embedding_view_thr_range.max_thr,
                                                                step=0.01, value=0.5, 
                                                                width=200)

        self._prepare_data_embedding_view('')
        self.graph_view_scatter = hv.DynamicMap(pn.bind(draw.draw_embedding_view_scatter,
                                                        layer=self.layer_selection,
                                                        colors=self.colors,),
            streams=[self.groups_stream,
                     self.data_embedding_view_xy,
                     self.data_embedding_view_alpha,
                     ])
        
        self.graph_view_scatter_selection1d = hv.streams.Selection1D(source=self.graph_view_scatter)

        # watch it
        self.graph_view_scatter_selection1d.add_subscriber(self._graph_view_scatter_selection1d_subscriber) 
        self.graph_view_polys = hv.DynamicMap(pn.bind(draw.draw_embedding_view_polys,
                                                      layer=self.layer_selection,
                                                      colors=self.colors,
                                                      threshold=self.embedding_view_thr_slider,),
            streams=[self.groups_stream,
                     self.data_embedding_view_xy,
                     self.data_embedding_view_polys,
                     ])
        
        self.graph_view_square = hv.DynamicMap(pn.bind(draw.draw_embedding_view_square,
                                                       layer=self.layer_selection,),
            streams=[
                     self.data_embedding_view_xy,
                     self.graph_view_scatter_selection1d
                     ])
        
        self.graph_view_overlay = (self.graph_view_square * self.graph_view_scatter * self.graph_view_polys).opts(
            width=int(self.node_selection_view_width*0.24) - self.line_space,
            height=int(self.node_selection_view_height*0.72),
            shared_axes=False,
        ) 
        
        self.graph_view_legend = hv.DynamicMap(pn.bind(draw.draw_embedding_view_legend,
                                                       colors=self.colors,),
            streams=[self.groups_stream]).opts(
                width=int(self.graph_view_width*0.2),
                height=int(self.graph_view_height*0.72),
            )

### fairness metric view
        self.fairness_metric_view_chart = pn.pane.HoloViews()
        self.fairness_metric_view_chart_column = pn.Column(
            '#### Detail of Selected Metric',
            self.fairness_metric_view_chart
        )

        self.fairness_metric_view_chart_eod_radio = pn.widgets.RadioButtonGroup(
            options=['TPR', 'FPR'], value='TPR', button_type='default', width=200)

        self.fairness_metric_view_value_bar = hv.DynamicMap(pn.bind(draw.draw_fairness_metric_view_value_bar,
                                                                     labels=self.labels,
                                                                     predictions=self.predictions),
                                                                streams=[self.selected_nodes_stream, 
                                                                         self.groups_stream,
                                                                        ])
        self.fairness_metric_view_range_bar = draw.draw_fairness_metric_view_range_bar()
        self.fairness_metric_view_bar = (self.fairness_metric_view_value_bar * self.fairness_metric_view_range_bar).opts(
            width=int(self.fairness_metric_view_width*0.93),
            height=int(self.fairness_metric_view_height*0.35),
        )
        self.fairness_metric_view_value_bar_selection1d = hv.streams.Selection1D(source=self.fairness_metric_view_value_bar)
        self.fairness_metric_view_value_bar_selection1d.add_subscriber(self._update_fairness_metric_detail)

        self.fairness_metric_view = pn.Card(
            pn.Column(
                '#### Fairness Metrics',
                self.fairness_metric_view_bar
            ), 
            pn.pane.HTML(f"<div style='height: 1px; background-color: {FILL_GREY_COLOR}; width: {int(self.fairness_metric_view_width)-20}px;'></div>"),
            self.fairness_metric_view_chart_column,
            hide_header=True,
            name='Fairness Metric View',
            height=self.fairness_metric_view_height,
            width=self.fairness_metric_view_width,
        )

### attribute view
        # overview
        feat_np = self.feat.cpu().numpy()
        self.columns_categorical = util.check_unique_values(feat_np)
        contributions = util.calc_contributions(
            model=self.model,
            g=self.g,
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups)
        self.contributions_stream = Contributions(contributions=contributions)

        self.groups_stream.add_subscriber(self._update_correlations_groups_callback)
        self.groups_stream.add_subscriber(self._groups_stream_subscriber)

        self.selected_attrs_ls_stream = SelectedAttrsLs(selected_attrs_ls=[[]])
        self.selected_attrs_ls_stream.add_subscriber(self._selected_attrs_ls_stream_callback)

        self.attribute_view_overview_tap = streams.SingleTap()
        self.attribute_view_overview_selected = hv.DynamicMap(pn.bind(draw.draw_attribute_view_overview_selected, 
                                                             feat=pd.DataFrame(self.feat.cpu().numpy(), columns=self.feat_names),
                                                             columns_categorical=self.columns_categorical,
                                                             hw_ratio=int(self.diagnostic_panel_height*0.5)/int(self.diagnostic_panel_width*0.93)
                                                             ),
                                                        streams=[self.selected_nodes_stream, 
                                                                 self.groups_stream,
                                                                 self.attribute_view_overview_tap,
                                                                 self.contributions_stream,
                                                                 self.selected_attrs_ls_stream,])
        # watch the tap
        self.attribute_view_overview_tap.add_subscriber(self._attribute_view_overview_tap_subscriber)

        self.attribute_view_overview_all = hv.DynamicMap(pn.bind(draw.draw_attribute_view_overview_all,
                                                             feat=pd.DataFrame(self.feat.cpu().numpy(), columns=self.feat_names),
                                                             columns_categorical=self.columns_categorical,
                                                             hw_ratio=int(self.diagnostic_panel_height*0.5)/int(self.diagnostic_panel_width*0.93)
                                                             ),
                                                        streams=[
                                                                 self.groups_stream,
                                                                 self.selected_attrs_ls_stream,])
        self.attribute_view_overview = (self.attribute_view_overview_all * self.attribute_view_overview_selected)

### dependency view
        self.dependency_view_attr_sens = pn.pane.HoloViews()
        self.dependency_view_attr_degree = pn.pane.HoloViews()

        computational_graph_degrees = []
        for i in range(self.max_hop):
            if i == 0:
                d = self.adj0.sum(axis=1).unsqueeze(1)
            else:
                d = torch.sparse.mm(self.adj0, d)
            computational_graph_degrees.append(d.to_dense().squeeze())
        # convert computational_graph_degrees to a np array
        self.computational_graph_degrees = torch.stack(computational_graph_degrees).cpu().numpy()  

        groups = self.groups_stream.groups
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adj = self.adj0
        # Group rows and sum them
        unique_groups = np.unique(groups)
        n_unique_groups = unique_groups.size

        group_connection_matrices = []
        max_hop = self.max_hop
        group_col_indices = []
        for k in range(max_hop):
            if k == 0:
                group_sums = torch.zeros((n_unique_groups, adj.shape[1])).to(device)
                # Iterate over each group, summing the corresponding rows
                for i, group in enumerate(unique_groups):
                    group_mask = torch.tensor(groups == group)  # Boolean array where True indicates the index belongs to group i
                    group_indices = torch.nonzero(group_mask).squeeze(-1).to(torch.long).to(device)  # Get the indices of the group
                    group_adj = torch.index_select(adj, 0, group_indices)
                    group_sums[i] = group_adj.sum(dim=0).to_dense()
            else:
                group_sums = torch.sparse.mm(group_sums, adj)

            group_connection_matrix = []
            # Iterate over each group to perform the extraction and summation of corresponding columns
            for i, group in enumerate(unique_groups):
                if k == 0:
                    group_mask = torch.tensor(groups == group)  # Boolean array where True indicates the index belongs to group i
                    group_indices = torch.nonzero(group_mask).squeeze(-1).to(torch.long).to(device)  # Get the indices of the group
                    group_col_indices.append(group_indices)
                else: 
                    group_indices = group_col_indices[i]
                group_group_sums = torch.index_select(group_sums, 1, group_indices)
                sum_group_group_sums = group_group_sums.sum(dim=1).to_dense()
                # convert sum_group_group_sums to a list
                sum_group_group_sums = sum_group_group_sums.tolist()
                for j, val in enumerate(sum_group_group_sums):
                    group_connection_matrix.append((unique_groups[i], unique_groups[j], val))
            group_connection_matrices.append(group_connection_matrix)

        self.group_connection_matrices_stream = GroupConnectionMatrices(group_connection_matrices=group_connection_matrices)

        self.dependency_view_structure_sens = hv.DynamicMap(pn.bind(draw.draw_dependency_view_structure_sens,
                                                                 hop=self.correlation_view_selection,
                                                                 scale=self.n_neighbors_scale_group,),
                                                                 streams=[self.group_connection_matrices_stream]).opts(
                                                                     height=int(self.diagnostic_panel_height*0.35) - self.line_space,
                                                                     width=int(self.diagnostic_panel_width*0.32),
                                                                    )

### density view
        # Slider for min_threshold
        avg_density = self.adj0._nnz() / (self.n_nodes * (self.n_nodes - 1))
        overall_density_string = f'Overall Density: {avg_density:.4f}'
        self.min_threshold_slider = pn.widgets.FloatSlider(name=f'Min Density Threshold', start=0.0, end=1.0, step=0.01, value=0.2, width=200)
        # v fast
        self.extract_communities_args = community.process_graph(self.adj0)
        # deep copy self.extract_communities_args, which is a tuple
        self.node_communities = community.extract_communities(self.extract_communities_args,
                                                              self.min_threshold_slider.value)
        self.communities = []
        for indices in self.node_communities:
            # Convert set of indices to sorted list for slicing
            indices_sorted = sorted(list(indices))
            # Slice rows: Efficient in CSR format
            row_sliced = self.adj0_scipy[indices_sorted, :]
            # Slice columns: Convert to CSC format for efficient column slicing if necessary
            # For simplicity, here we use the CSR format (less efficient for cols)
            final_slice = row_sliced[:, indices_sorted]
            self.communities.append(final_slice)
        graph_metrics = util.calculate_graph_metrics(self.communities) # Adjust this line as per your actual data structure
        self.data_density_view = DataDensityView(graph_metrics=graph_metrics)
        self.density_view_scatter = hv.DynamicMap(draw.draw_density_view_scatter,
                                                  streams=[self.data_density_view]).opts(
                  width=int(self.node_selection_view_width*0.3) - self.line_space, 
                  height=int(self.node_selection_view_height*0.75)-60,
                  shared_axes=False,
                  )
        
        self.density_view_scatter_selection1d = hv.streams.Selection1D(source=self.density_view_scatter)
        # watch it
        self.density_view_scatter_selection1d.add_subscriber(self._update_selected_communities_dropdown)

        self.selected_communities_dropdown = pn.widgets.Select(name='Clicked Subgraphs', options=[None],
                                                               width=int(self.correlation_view_width*0.93) - self.line_space)
        # watch it
        self.selected_communities_dropdown.param.watch(self._selected_communities_dropdown_callback, 'value')

        # Button to recalculate and redraw plots
        min_threshold_slider_button = pn.widgets.Button(name='Update')

        min_threshold_slider_button.on_click(self._min_threshold_slider_button_callback)
        
### diagnostic panel
        self.attr_selection_mode_button = pn.widgets.RadioButtonGroup(
            name='Selection Mode',
            options=['Single', 'Multiple'],
            value='Single',
            width=130,  
        )

        self.new_selection_button = pn.widgets.Button(name='New Selection', width=130)
        self.new_selection_button.on_click(self._new_selection_button_callback)

        contribution_attrs = individual_bias.calc_attr_contributions(
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
            attr_indices=list(range(self.n_feat))
        )
        contribution_structure = individual_bias.calc_structure_contribution(
            adj=self.adj,
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
        )
        contribution_attrs_structure = individual_bias.calc_attr_structure_contribution(
            adj=self.adj,
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
            attr_indices=list(range(self.n_feat))
        )
        contribution_attrs_summative = self.contributions_stream.contributions.sum()

        self.bias_contributions_attrs_summative_latex = pn.pane.LaTeX(f'Attr.(Summative): {contribution_attrs_summative:.3f}')
        self.bias_contributions_attrs_synergistic_latex = pn.pane.LaTeX(f'Attr.(Synergistic): {contribution_attrs:.3f}')
        self.bias_contributions_structure_latex = pn.pane.LaTeX(f'Struc.: {contribution_structure:.3f}')
        self.bias_contributions_attrs_structure_latex = pn.pane.LaTeX(f'Attr. & Struc.: {contribution_attrs_structure:.3f}')

        self.attribute_view_overview.opts(
            height=int(self.diagnostic_panel_height*0.41),
            width=int(self.diagnostic_panel_width*0.83) - 3*self.line_space,
        )
        self.diagnostic_panel = pn.Card(
            pn.Row(
                pn.Column(
                    '#### Bias Contributions',
                    self.bias_contributions_attrs_summative_latex,
                    self.bias_contributions_attrs_synergistic_latex,
                    self.bias_contributions_structure_latex,
                    self.bias_contributions_attrs_structure_latex,
                ),
                pn.pane.HTML(f"<div style='width: 1px; background-color: {FILL_GREY_COLOR}; height: {int(self.diagnostic_panel_height*0.5)}px;'></div>"),
                pn.Column(
                    pn.Row(
                        '#### Attr. Overview',
                        pn.Row(
                            pn.pane.LaTeX('Selection Mode', styles={'margin-top': '11px'}), 
                            self.attr_selection_mode_button
                        ),
                        self.new_selection_button,
                    ),
                    self.attribute_view_overview,
                ),
            ),
            pn.pane.HTML(f"<div style='width: {int(self.diagnostic_panel_width)-20}px; background-color: {FILL_GREY_COLOR}; height: 1px;'></div>"),
            pn.Row(
                pn.Column(
                    '#### Connectivity between Sensitive Groups',
                    self.dependency_view_structure_sens
                ),
                pn.pane.HTML(f"<div style='width: 1px; background-color: {FILL_GREY_COLOR}; height: {int(self.diagnostic_panel_height*0.42)}px;'></div>"),
                pn.Column(
                    '#### Attr. Distribution in each Sensitive Group',
                    self.dependency_view_attr_sens,
                    width=int(self.diagnostic_panel_width*0.3),
                ),
                pn.pane.HTML(f"<div style='width: 1px; background-color: {FILL_GREY_COLOR}; height: {int(self.diagnostic_panel_height*0.42)}px;'></div>"),
                pn.Column(
                    '#### Relationship between Attr. & # of Neighbors',
                    self.dependency_view_attr_degree,
                    width=int(self.diagnostic_panel_width*0.3) - self.line_space,
                ),
            ),
            hide_header=True,
            name='Diagnostic View',
            height=self.diagnostic_panel_height,
            width=self.diagnostic_panel_width,
        )

### node selection view
        # original scale
        degree_hist_frequencies = []
        self.degree_hist_edges = []
        for i in range(self.max_hop):
            d = self.computational_graph_degrees[i]
            frequencies, edges = np.histogram(d, 20)
            degree_hist_frequencies.append(frequencies)
            self.degree_hist_edges.append(edges)
        # log scale
        degree_hist_frequencies_log = []
        self.degree_hist_edges_log = []
        for i in range(self.max_hop):
            d = self.computational_graph_degrees[i]
            frequencies, edges = np.histogram(np.log(d+1), 20)
            degree_hist_frequencies_log.append(frequencies)
            self.degree_hist_edges_log.append(edges)
        frequencies_dict = {'Original': degree_hist_frequencies, 'Log': degree_hist_frequencies_log}
        edges_dict = {'Original': self.degree_hist_edges, 'Log': self.degree_hist_edges_log}
        self.structural_bias_overview_hist_all = hv.DynamicMap(pn.bind(draw.draw_structural_bias_overview_hist_all,
                                                                       frequencies_dict=frequencies_dict,
                                                                       edges_dict=edges_dict,
                                                                       hop=self.correlation_view_selection,
                                                                       scale=self.n_neighbors_scale_group),
                                                              )
        self.structural_bias_overview_hist_selected = hv.DynamicMap(pn.bind(draw.draw_structural_bias_overview_hist_selected,
                                                                            computational_graph_degrees=self.computational_graph_degrees,
                                                                            edges_dict=edges_dict,
                                                                            hop=self.correlation_view_selection,
                                                                            scale=self.n_neighbors_scale_group),
                                                                     streams=[self.pre_selected_nodes_stream])
        self.structural_bias_overview_hist = (self.structural_bias_overview_hist_selected * self.structural_bias_overview_hist_all).opts(
            width=int(self.node_selection_view_width*0.3) - self.line_space,
            height=int(self.node_selection_view_height*0.73),  
        )

        self.structural_bias_overview_hist_all_selection1d = hv.streams.Selection1D(source=self.structural_bias_overview_hist_all)
        self.structural_bias_overview_hist_all_selection1d.add_subscriber(self._structural_bias_overview_hist_all_selection1d_subscriber)

        self.node_selectin_view = pn.Card(
            pn.Row(
                pn.Column(
                    '#### Node Embeddings',
                    pn.Row(
                        self.graph_view_legend,   
                        self.graph_view_overlay,
                    )
                ),
                pn.pane.HTML(f"<div style='width: 1px; background-color: {FILL_GREY_COLOR}; height: {self.node_selection_view_height-20}px;'></div>"),
                pn.Column(
                    '#### # of Neighbors in Computational Graphs',
                    self.structural_bias_overview_hist,
                ),
                pn.pane.HTML(f"<div style='width: 1px; background-color: {FILL_GREY_COLOR}; height: {self.node_selection_view_height-20}px;'></div>"),
                pn.Column(
                    '#### Dense Subgraphs',
                    self.selected_communities_dropdown,
                    self.density_view_scatter,
                ),
            ),
            hide_header=True,
            name='Node Selection View',
            height=self.node_selection_view_height,
            width=self.node_selection_view_width, 
        )

# control panel
        record_nodes_selector_options = list(np.unique(self.groups_stream.groups))
        self.record_nodes_selector = pn.widgets.MultiChoice(name='Nodes', 
                                                     options=record_nodes_selector_options, 
                                                     value=record_nodes_selector_options, 
                                                     width=int(self.control_panel_width*0.63))  
         
        record_attribute_selection_options = ['All', 'None']
        selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls
        if [] in selected_attrs_ls:
            record_attribute_selection_options += list(range(1, len(selected_attrs_ls)))
        else:
            record_attribute_selection_options += list(range(1, len(selected_attrs_ls)+1))
        self.record_attribute_selection = pn.widgets.Select(options=record_attribute_selection_options, 
                                                            name='Attributes', 
                                                            value='All', 
                                                            width=200)
        
        self.record_edge_switch = pn.widgets.Switch(name='Edges', value=True)

        self.control_panel = pn.Card(
            pn.Column(
                '### Global Settings', 
                '#### Sensitive Attribute Selection',
                pn.Row(
                    self.sens_name_selector, 
                    sens_name_confirm_control_panel_button,
                ), 
                '### Hop Selection',
                self.layer_selection, 
                '#### Node Selection',
                self.n_selected_nodes_latex,
                pn.Row(
                    node_selection_confirm_control_panel_button,
                    node_selection_clear_control_panel_button,
                ),
                '#### Record',
                self.record_nodes_selector,
                self.record_attribute_selection,
                pn.Row(pn.pane.Str('Edges: False'), self.record_edge_switch, pn.pane.Str('True')),
                self.record_button,
                '#### Scale of # of Neighbors',
                self.n_neighbors_scale_group,
                pn.layout.Divider(), 
                '### Node Selection View Settings',
                '#### Node Embeddings Settings',
                self.projection_selection,
                pn.Row(
                    self.node_sample_size_slider,
                    sample_button,
                ),
                self.embedding_view_thr_slider,
                '#### Dense Subgraphs Settings',
                overall_density_string,
                pn.Row(
                    self.min_threshold_slider, 
                    min_threshold_slider_button,
                ),
                scroll=True,
            ),
            name='Control Panel',
            hide_header=True,  
            height=self.control_panel_height, 
            width=self.control_panel_width,
        ) 

        app = pn.GridSpec(width=self.width, height=self.height)
        app[0, 0] = pn.Tabs(
            self.control_panel,
        )
        app[1: 3, 0] = pn.Tabs(
            self.fairness_metric_view,
        )
        app[0, 1: 4] = pn.Tabs(
            self.node_selectin_view
        )
        app[1: 3, 1: 4] = pn.Tabs(
            self.diagnostic_panel,
        )

        return app

    def _sample_button_callback(self, event):
        node_sample_size = self.node_sample_size_slider.value
        if node_sample_size < self.n_nodes:
            sampled_alpha = np.zeros(self.n_nodes)
            sample_idx = np.random.choice(np.arange(self.n_nodes), node_sample_size, replace=False)
            sampled_alpha[sample_idx] = 1
            self.sampled_alpha = sampled_alpha
        else:
            self.sampled_alpha = np.ones(self.n_nodes)
        
        selected_nodes = self.pre_selected_nodes_stream.selected_nodes.astype(int)
        if len(selected_nodes) > 0:
            # create a np array containing 1 and 0.2 representing the alpha value of the nodes selected and not selected
            selected_alpha = np.zeros(self.n_nodes)
            selected_alpha[selected_nodes] = 1
            # not selected nodes are set to 0
            selected_alpha[selected_alpha == 0] = 0
            alpha = selected_alpha * self.sampled_alpha 
        else:
            alpha = self.sampled_alpha

        self.data_embedding_view_alpha.event(alpha=alpha)

    def _sens_name_confirm_control_panel_button_callback(self, event):
        if self.sens_name_selector.value != self.previous_sens_name_value:
            # Find the indices of the selected sensitive attributes
            sens_indices = [self.sens_names.index(name) for name in self.sens_name_selector.value]
            sens_selected = self.sens[sens_indices]
            vectorized_concat = np.vectorize(util.concatenate_elements)
            # Apply the vectorized function across columns
            sens = vectorized_concat(*sens_selected)
            self.groups_stream.event(groups=sens)

        self.previous_sens_name_value = self.sens_name_selector.value

    def _attribute_view_overview_tap_subscriber(self, x, y):
        x, y = y, x
        if not (self.previous_attribute_view_overview_tap_x == x and self.previous_attribute_view_overview_tap_y == y):
            if y:
                # get the data of self.attribute_view_overview
                df = self.attribute_view_overview_selected.data[()].get('Rectangles.I').data
                # find the corresponding id satisfying the y0 <= y <= y1
                id = df[(df['y0'] <= y) & (df['y1'] >= y)]['ID']
                if self.attr_selection_mode_button.value == 'Single':
                    if id.empty:
                        self.dependency_view_attr_sens.object = None
                        self.dependency_view_attr_degree.object = None
                    else:
                        id = id.values[0]
                        feat = self.feat[:, id].cpu().numpy()
                        feat_name = self.feat_names[id]
                        if self.columns_categorical[id]:
                            dependency_view_attr_sens_bar_all = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_sens_bar_all,
                                                                                      variable_data=feat,
                                                                                      feat_name=feat_name,),
                                                                               streams=[self.groups_stream])
                            dependency_view_attr_sens_bar_selected = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_sens_bar_selected,
                                                                                             variable_data=feat,
                                                                                             feat_name=feat_name,),
                                                                                        streams=[self.groups_stream,
                                                                                                self.selected_nodes_stream])
                            self.dependency_view_attr_sens.object = (dependency_view_attr_sens_bar_selected * dependency_view_attr_sens_bar_all).opts(
                                height=int(self.diagnostic_panel_height*0.34),
                                width=int(self.diagnostic_panel_width*0.3) - self.line_space,
                                ) 
                            
                            self.dependency_view_attr_degree.object = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_degree_violin,
                                                                                            feat=feat,
                                                                                            computational_graph_degrees=self.computational_graph_degrees,
                                                                                            hop=self.correlation_view_selection,
                                                                                            scale=self.n_neighbors_scale_group,),
                                                                                    streams=[self.selected_nodes_stream,
                                                                                    ]
                                                                                    ).opts(
                                                                                        height=int(self.diagnostic_panel_height*0.34),
                                                                                        width=int(self.diagnostic_panel_width*0.3) - self.line_space,
                                                                                    )
                        else:
                            self.dependency_view_attr_sens.object = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_sens_violin,
                                                                                   feat=feat),
                                                                           streams=[self.selected_nodes_stream,
                                                                                    self.groups_stream]).opts(
                                                                                        height=int(self.diagnostic_panel_height*0.34),
                                                                                        width=int(self.diagnostic_panel_width*0.3) - self.line_space,
                                                                                    )
                            
                            dependency_view_attr_degree_hex_all = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_degree_hex_all,
                                                                                        feat=feat,
                                                                                        computational_graph_degrees=self.computational_graph_degrees,
                                                                                        hop=self.correlation_view_selection,
                                                                                        scale=self.n_neighbors_scale_group,),
                                                                                        )
                            dependency_view_attr_degree_scatter_selected = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_degree_scatter_selected,
                                                                                             feat=feat,
                                                                                             computational_graph_degrees=self.computational_graph_degrees,
                                                                                             hop=self.correlation_view_selection,
                                                                                             scale=self.n_neighbors_scale_group,),
                                                                                     streams=[self.data_embedding_view_alpha,
                                                                                              self.selected_nodes_stream,]) 
                            self.dependency_view_attr_degree.object = (dependency_view_attr_degree_hex_all * dependency_view_attr_degree_scatter_selected).opts(
                                height=int(self.diagnostic_panel_height*0.34),
                                width=int(self.diagnostic_panel_width*0.3) - self.line_space,
                                shared_axes=False
                            )
                else:
                    self.dependency_view_attr_sens.object = None
                    self.dependency_view_attr_degree.object = None
                    if not id.empty:
                        id = id.values[0]
                        if id in self.selected_attrs_ls_stream.selected_attrs_ls[-1]:
                            selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
                            selected_attrs_ls[-1].remove(id)
                            self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
                        else:
                            selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
                            selected_attrs_ls[-1].append(id)
                            self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
                
                # calculate the deleting area
                selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
                circle_gap = 1
                r_sector1 = 0.45
                n_selected_attrs = len(selected_attrs_ls)
                n_all_nodes = self.n_nodes
                n = self.n_feat
                hw_ratio = int(self.diagnostic_panel_height*0.5)/int(self.diagnostic_panel_width*0.93)
                ymax = n+1.5
                ymin = 0 
                xmax = n_all_nodes * 1.1 
                xmin = (circle_gap * xmax * (n_selected_attrs + 1)) / (circle_gap * n_selected_attrs + circle_gap - hw_ratio * ymax + hw_ratio * ymin)
                if abs(xmin) > xmax:
                    # solve: -xmax = -circle_gap * (1 + n_selected_attrs) * (2 * xmax / (ymax - ymin)) / hw_ratio
                    circle_gap = (hw_ratio * (ymax - ymin)) / (2 * (1 + n_selected_attrs))
                    xmin = -xmax
                x_y_ratio = (xmax - xmin) / (ymax - ymin) / hw_ratio 
                y_pos = n + 0.5 
                for i, selected_attrs in enumerate(selected_attrs_ls): 
                    x_pos = -circle_gap * (0.5 + i + 1) * x_y_ratio
                    if selected_attrs:
                        x_left = x_pos - r_sector1 * x_y_ratio 
                        x_right = x_pos + r_sector1 * x_y_ratio
                        y_bottom = y_pos - r_sector1 
                        y_top = y_pos + r_sector1
                        # check if (x, y) is in the deleting area
                        if x_left <= x <= x_right and y_bottom <= y <= y_top:
                            selected_attrs_ls.pop(i)
                            if selected_attrs_ls:
                                self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
                            else: 
                                self.selected_attrs_ls_stream.event(selected_attrs_ls=[[]])
                            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
                            tmp_contributions_selected_attrs = np.delete(contributions_selected_attrs, i)
                            if tmp_contributions_selected_attrs.size == 0:
                                self.contributions_stream.event(contributions_selected_attrs=np.array([0.]))
                            else:
                                self.contributions_stream.event(contributions_selected_attrs=tmp_contributions_selected_attrs)
                            break
        self.previous_attribute_view_overview_tap_x, self.previous_attribute_view_overview_tap_y = x, y

    def _update_selected_communities_dropdown(self, index): 
        if index:
            self.selected_communities_dropdown.options = [None] + [str(list(self.node_communities[i]))[1:-1] for i in index]
        else:
            self.selected_communities_dropdown.options = [None] 
        self.selected_communities_dropdown.value = self.selected_communities_dropdown.options[0]

    def _selected_communities_dropdown_callback(self, event):
        if self.selected_communities_dropdown.value:
            # convert node_indices_selected to a list
            selected_nodes = np.array(list(map(int, self.selected_communities_dropdown.value.split(', '))))
        else:
            selected_nodes = np.array([])
        pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
        self.pre_selected_nodes_stream.event(selected_nodes=np.union1d(pre_selected_nodes, selected_nodes))

    def _update_fairness_metric_detail(self, index):
        if index:
            metric_name = index[0]
            chart = hv.DynamicMap(pn.bind(draw.draw_fairness_metric_view_detail,
                                        metric_name=metric_name,
                                        labels=self.labels,
                                        predictions=self.predictions,
                                        eod_radio=self.fairness_metric_view_chart_eod_radio),
                                streams=[self.selected_nodes_stream, self.groups_stream]) 

            if metric_name == 2 or metric_name == 6:
                if len(self.fairness_metric_view_chart_column) == 2:
                    self.fairness_metric_view_chart_column.insert(1, self.fairness_metric_view_chart_eod_radio)
                    chart.opts(width=int(self.fairness_metric_view_width*0.93), height=int(self.fairness_metric_view_height*0.33)) 
            else:  
                if len(self.fairness_metric_view_chart_column) == 3: 
                    self.fairness_metric_view_chart_column.pop(1)   
                chart.opts(width=int(self.fairness_metric_view_width*0.93), height=int(self.fairness_metric_view_height*0.39)) 

            self.fairness_metric_view_chart.object = chart
        else:
            self.fairness_metric_view_chart.object = None

    def _min_threshold_slider_button_callback(self, event):
        thr = self.min_threshold_slider.value
        # Recalculate communities
        self.node_communities = community.extract_communities(self.extract_communities_args,
                                                                thr)
        self.communities = []
        for indices in self.node_communities:
            # Convert set of indices to sorted list for slicing
            indices_sorted = sorted(list(indices))
            # Slice rows: Efficient in CSR format
            row_sliced = self.adj0_scipy[indices_sorted, :]
            # Slice columns: Convert to CSC format for efficient column slicing if necessary
            # For simplicity, here we use the CSR format (less efficient for cols)
            final_slice = row_sliced[:, indices_sorted]
            self.communities.append(final_slice)
        graph_metrics = util.calculate_graph_metrics(self.communities)
        self.data_density_view.event(graph_metrics=graph_metrics)

    def _pre_selected_nodes_stream_subscriber(self, selected_nodes):
        if len(selected_nodes) > 0:
            selected_alpha = np.zeros(self.n_nodes)
            selected_alpha[self.pre_selected_nodes_stream.selected_nodes.astype(int)] = 1
            # element-wise multiplication of the selected_alpha and the sampled_alpha
            alpha = selected_alpha * self.sampled_alpha
        else:
            alpha = self.sampled_alpha
        # update the alpha of the nodes
        self.data_embedding_view_alpha.event(alpha=alpha)

        # update self.n_selected_nodes_latex
        self.n_selected_nodes_latex.object = f'# of Selected Nodes: {len(selected_nodes)}/{self.n_nodes} ({(len(selected_nodes) / self.n_nodes * 100):.2f}%)'

    def _selected_nodes_stream_subscriber(self, selected_nodes):
        # update contributions
        contributions = util.calc_contributions(
            model=self.model,
            g=self.g,
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups)
        
        selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls
        contributions_selected_attrs = []
        for selected_attrs in selected_attrs_ls:
            if selected_attrs:
                contribution_selected_attrs = individual_bias.calc_attr_contributions(
                    model=self.model,
                    g=self.g, 
                    feat=self.feat,
                    selected_nodes=self.selected_nodes_stream.selected_nodes,
                    groups=self.groups_stream.groups,
                    attr_indices=selected_attrs
                )
                contributions_selected_attrs.append(contribution_selected_attrs)
            else:
                contributions_selected_attrs.append(0.)
        self.contributions_stream.event(contributions=contributions,
                                        contributions_selected_attrs=np.array(contributions_selected_attrs))

        contribution_attrs = individual_bias.calc_attr_contributions(
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
            attr_indices=list(range(self.n_feat))
        )
        contribution_structure = individual_bias.calc_structure_contribution(
            adj=self.adj,
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
        )
        contribution_attrs_structure = individual_bias.calc_attr_structure_contribution(
            adj=self.adj,
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
            attr_indices=list(range(self.n_feat))
        )
        contribution_attrs_summative = self.contributions_stream.contributions.sum()
        self.bias_contributions_attrs_summative_latex.object = f'Attr.(Summative): {contribution_attrs_summative:.3f}'
        self.bias_contributions_attrs_synergistic_latex.object = f'Attr.(Synergistic): {contribution_attrs:.3f}'
        self.bias_contributions_structure_latex.object = f'Structure: {contribution_structure:.3f}'
        self.bias_contributions_attrs_structure_latex.object = f'Attr. & Struc.: {contribution_attrs_structure:.3f}'

        # update self.group_connection_matrices_stream
        groups = self.groups_stream.groups
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adj = self.adj0
        # Group rows and sum them
        unique_groups = np.unique(groups)
        n_unique_groups = unique_groups.size
        groups_selected = groups[selected_nodes.astype(int)]
        # Convert selected_nodes to tensor for advanced indexing
        selected_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long, device=device)
        # Filter rows from adj
        filtered_adj = torch.index_select(adj, 0, selected_nodes_tensor)

        group_connection_matrices = []
        max_hop = self.max_hop
        group_col_indices = []
        for k in range(max_hop):
            # row operation
            if k == 0:
                group_sums = torch.zeros((n_unique_groups, adj.shape[1])).to(device)
                # Iterate over each group, summing the corresponding rows
                for i, group in enumerate(unique_groups):
                    group_mask = torch.tensor(groups_selected == group)  # Boolean array where True indicates the index belongs to group i
                    group_indices = torch.nonzero(group_mask).squeeze(-1).to(torch.long).to(device)  # Get the indices of the group
                    group_adj = torch.index_select(filtered_adj, 0, group_indices)
                    group_sums[i] = group_adj.sum(dim=0).to_dense()
            else:
                group_sums = torch.sparse.mm(group_sums, adj)
            # col operation
            group_connection_matrix = []
            # Iterate over each group to perform the extraction and summation of corresponding columns
            for i, group in enumerate(unique_groups):
                if k == 0:
                    group_mask = torch.tensor(groups == group)  # Boolean array where True indicates the index belongs to group i
                    group_indices = torch.nonzero(group_mask).squeeze(-1).to(torch.long).to(device)  # Get the indices of the group
                    group_col_indices.append(group_indices)
                else: 
                    group_indices = group_col_indices[i]
                group_group_sums = torch.index_select(group_sums, 1, group_indices)
                sum_group_group_sums = group_group_sums.sum(dim=1).to_dense()
                # convert sum_group_group_sums to a list
                sum_group_group_sums = sum_group_group_sums.tolist()
                for j, val in enumerate(sum_group_group_sums):
                    group_connection_matrix.append((unique_groups[i], unique_groups[j], val))
            group_connection_matrices.append(group_connection_matrix)

        self.group_connection_matrices_stream.event(group_connection_matrices=group_connection_matrices)

    def _groups_stream_subscriber(self, groups):
        # update contributions
        contributions = util.calc_contributions(
            model=self.model,
            g=self.g,
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups)
        
        selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls
        contributions_selected_attrs = []
        for selected_attrs in selected_attrs_ls:
            if selected_attrs:
                contribution_selected_attrs = individual_bias.calc_attr_contributions(
                    model=self.model,
                    g=self.g, 
                    feat=self.feat,
                    selected_nodes=self.selected_nodes_stream.selected_nodes,
                    groups=self.groups_stream.groups,
                    attr_indices=selected_attrs
                )
                contributions_selected_attrs.append(contribution_selected_attrs)
            else:
                contributions_selected_attrs.append(0.)
        self.contributions_stream.event(contributions=contributions,
                                        contributions_selected_attrs=np.array(contributions_selected_attrs))

        contribution_attrs = individual_bias.calc_attr_contributions(
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
            attr_indices=list(range(self.n_feat))
        )
        contribution_structure = individual_bias.calc_structure_contribution(
            adj=self.adj,
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
        )
        contribution_attrs_structure = individual_bias.calc_attr_structure_contribution(
            adj=self.adj,
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
            attr_indices=list(range(self.n_feat))
        )
        contribution_attrs_summative = self.contributions_stream.contributions.sum()
        self.bias_contributions_attrs_summative_latex.object = f'Attr.(Summative): {contribution_attrs_summative:.3f}'
        self.bias_contributions_attrs_synergistic_latex.object = f'Attr.(Synergistic): {contribution_attrs:.3f}'
        self.bias_contributions_structure_latex.object = f'Structure: {contribution_structure:.3f}'
        self.bias_contributions_attrs_structure_latex.object = f'Attr. & Struc.: {contribution_attrs_structure:.3f}'

        self.record_nodes_selector.options = list(np.unique(groups))
        self.record_nodes_selector.value = self.record_nodes_selector.options

        # update self.group_connection_matrices_stream
        selected_nodes = self.selected_nodes_stream.selected_nodes
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        adj = self.adj0
        # Group rows and sum them
        unique_groups = np.unique(groups)
        n_unique_groups = unique_groups.size
        groups_selected = groups[selected_nodes.astype(int)]
        # Convert selected_nodes to tensor for advanced indexing
        selected_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long, device=device)
        # Filter rows from adj
        filtered_adj = torch.index_select(adj, 0, selected_nodes_tensor)

        group_connection_matrices = []
        max_hop = self.max_hop
        group_col_indices = []
        for k in range(max_hop):
            # row operation
            if k == 0:
                group_sums = torch.zeros((n_unique_groups, adj.shape[1])).to(device)
                # Iterate over each group, summing the corresponding rows
                for i, group in enumerate(unique_groups):
                    group_mask = torch.tensor(groups_selected == group)  # Boolean array where True indicates the index belongs to group i
                    group_indices = torch.nonzero(group_mask).squeeze(-1).to(torch.long).to(device)  # Get the indices of the group
                    group_adj = torch.index_select(filtered_adj, 0, group_indices)
                    group_sums[i] = group_adj.sum(dim=0).to_dense()
            else:
                group_sums = torch.sparse.mm(group_sums, adj)
            # col operation
            group_connection_matrix = []
            # Iterate over each group to perform the extraction and summation of corresponding columns
            for i, group in enumerate(unique_groups):
                if k == 0:
                    group_mask = torch.tensor(groups == group)  # Boolean array where True indicates the index belongs to group i
                    group_indices = torch.nonzero(group_mask).squeeze(-1).to(torch.long).to(device)  # Get the indices of the group
                    group_col_indices.append(group_indices)
                else: 
                    group_indices = group_col_indices[i]
                group_group_sums = torch.index_select(group_sums, 1, group_indices)
                sum_group_group_sums = group_group_sums.sum(dim=1).to_dense()
                # convert sum_group_group_sums to a list
                sum_group_group_sums = sum_group_group_sums.tolist()
                for j, val in enumerate(sum_group_group_sums):
                    group_connection_matrix.append((unique_groups[i], unique_groups[j], val))
            group_connection_matrices.append(group_connection_matrix)

        self.group_connection_matrices_stream.event(group_connection_matrices=group_connection_matrices)        

    def _update_correlations_groups_callback(self, groups): 
        # update contributions
        contributions = util.calc_contributions(
            model=self.model,
            g=self.g,
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups)
        self.contributions_stream.event(contributions=contributions)        

    def _projection_selection_callback(self, event):
        projection_name = self.projection_selection.value
        if projection_name == 'UMAP':
            # check if self.embeddings_umap is an empty list
            if not self.embeddings_umap:
                # compute the embeddings
                self.embeddings_umap = util.proj_emb(self.embeddings, 'umap')
            # update the embeddings
            self.data_embedding_view_xy.event(xy=np.array(self.embeddings_umap))
        elif projection_name == 't-SNE':
            # check if self.embeddings_tsne is an empty list
            if not self.embeddings_tsne:
                # compute the embeddings
                self.embeddings_tsne = util.proj_emb(self.embeddings, 'tsne')
            # update the embeddings
            self.data_embedding_view_xy.event(xy=np.array(self.embeddings_tsne))
        elif projection_name == 'PCA':
            # check if self.embeddings_pca is an empty list
            if not self.embeddings_pca:
                # compute the embeddings
                self.embeddings_pca = util.proj_emb(self.embeddings, 'pca')
            # update the embeddings
            self.data_embedding_view_xy.event(xy=np.array(self.embeddings_pca))

    def _prepare_data_embedding_view(self, event=None, xy=None, groups=None):
        layer = self.layer_selection.value - 1
        self.correlation_view_selection.value = layer
        
        if (tuple(self.sens_name_selector.value), layer) not in self.pre_compute_contours_cache:
            xy = self.data_embedding_view_xy.xy
            groups = self.groups_stream.groups
            sens_selected = groups
            sens_selected_unique = np.unique(sens_selected)
            colormap = {sens_selected_unique[i]: self.colors[i] for i in range(len(sens_selected_unique))}

            min_thr, max_thr, polygons_lst, max_edges_lst = RangesetCategorical.pre_compute_contours(colormap, 
                                                                                                 xy, layer, groups)
            self.pre_compute_contours_cache[(tuple(self.sens_name_selector.value), layer)] = (min_thr, max_thr, polygons_lst, max_edges_lst)
        else:
            min_thr, max_thr, polygons_lst, max_edges_lst = self.pre_compute_contours_cache[(tuple(self.sens_name_selector.value), layer)]
        # update polygons_lst, max_edges_lst in the data_embedding_view
        self.data_embedding_view_polys.event(polygons_lst=polygons_lst, max_edges_lst=max_edges_lst)
        self.data_embedding_view_thr_range.event(min_thr=min_thr, max_thr=max_thr)

    def _update_thr_range(self, min_thr, max_thr):
        self.embedding_view_thr_slider.start = self.data_embedding_view_thr_range.min_thr
        self.embedding_view_thr_slider.end = self.data_embedding_view_thr_range.max_thr
        self.embedding_view_thr_slider.value = (self.data_embedding_view_thr_range.max_thr - self.data_embedding_view_thr_range.min_thr) / 10

    def _selected_attrs_ls_stream_callback(self, selected_attrs_ls):
        if selected_attrs_ls[-1]:
            # update contributions_stream.contributions_selected_attrs[-1] according to the selected_attrs_ls[-1]
            contribution_selected_attrs = individual_bias.calc_attr_contributions(
                model=self.model,
                g=self.g, 
                feat=self.feat,
                selected_nodes=self.selected_nodes_stream.selected_nodes,
                groups=self.groups_stream.groups,
                attr_indices=selected_attrs_ls[-1]
            )
            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
            contributions_selected_attrs[-1] = contribution_selected_attrs
        else:
            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
            contributions_selected_attrs[-1] = 0.

        self.contributions_stream.event(contributions_selected_attrs=contributions_selected_attrs) 

        record_attribute_selection_options = ['All', 'None']
        selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls
        if [] in selected_attrs_ls:
            record_attribute_selection_options += list(range(1, len(selected_attrs_ls)))
        else:
            record_attribute_selection_options += list(range(1, len(selected_attrs_ls)+1))
        self.record_attribute_selection.options = record_attribute_selection_options
        self.record_attribute_selection.value = 'All'

    def _new_selection_button_callback(self, event):
        contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy()
        # create a new list of selected attributes
        selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
        selected_attrs_ls.append([])
        self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
        # append a 0. to self.contributions_stream.contributions_selected_attrs
        contributions_selected_attrs_tmp = np.append(contributions_selected_attrs, 0.) 
        self.contributions_stream.event(contributions_selected_attrs=contributions_selected_attrs_tmp)

    def _record_button_callback(self, event):
        selected_nodes = self.selected_nodes_stream.selected_nodes.astype(int)
        # filter the selected_nodes according to self.record_nodes_selector, use vectorized function
        # Convert list to numpy array for efficient comparison
        selector_values = np.array(self.record_nodes_selector.value)
        # Create a boolean mask where True means the node's group is in selector_values
        mask = np.isin(self.groups_stream.groups[selected_nodes], selector_values)
        # Apply mask to selected_nodes to filter it
        recorded_nodes = selected_nodes[mask]

        if self.record_attribute_selection.value == 'All':
            recorded_attrs = self.feat_names
        elif self.record_attribute_selection.value == 'None':
            recorded_attrs = []
        else:
            idx = self.record_attribute_selection.value - 1
            selected_attrs = self.selected_attrs_ls_stream.selected_attrs_ls[idx]
            recorded_attrs = [self.feat_names[i] for i in selected_attrs]

        self.records.append({
            'Sens': self.sens_name_selector.value,
            'Nodes': recorded_nodes, 
            'Edges': self.record_edge_switch.value,
            'Attributes': recorded_attrs,
        })   

    def _graph_view_scatter_selection1d_subscriber(self, index):
        # device of self.adj0
        device = self.adj0.device
        if index:
            if len(index) > 1:
                pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
                self.pre_selected_nodes_stream.event(selected_nodes=np.union1d(pre_selected_nodes, np.array(index)))
            else:
                idx = index[0]
                selected_nodes = torch.tensor([idx]).to(device)
                for i in range(self.max_hop):
                    if i == 0:
                        neigh_i = self.adj0[idx].unsqueeze(0)
                    else:
                        neigh_i = torch.mm(neigh_i, self.adj0)
                    selected_nodes = torch.cat((selected_nodes, neigh_i.coalesce().indices()[1]))
                selected_nodes = torch.unique(selected_nodes).cpu().numpy().astype(int)
                pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
                self.pre_selected_nodes_stream.event(selected_nodes=np.union1d(pre_selected_nodes, selected_nodes))
                
        else:
            pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
            self.pre_selected_nodes_stream.event(selected_nodes=np.union1d(pre_selected_nodes, np.array([])))

    def get_records(self):
        return self.records

    def _node_selection_confirm_control_panel_button_callback(self, event): 
        pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
        self.selected_nodes_stream.event(selected_nodes=pre_selected_nodes)  

    def _node_selection_clear_control_panel_button_callback(self, event):
        self.pre_selected_nodes_stream.event(selected_nodes=np.array([]))
        if len(self.selected_nodes_stream.selected_nodes) < self.n_nodes:
            self.selected_nodes_stream.event(selected_nodes=self.node_indices)
        
        self.selected_communities_dropdown.value = None

    def _structural_bias_overview_hist_all_selection1d_subscriber(self, index):
        if index:
            layer = self.correlation_view_selection.value
            if self.n_neighbors_scale_group.value == 'Original':
                min_degree = self.degree_hist_edges[layer][index[0]]
                max_degree = self.degree_hist_edges[layer][index[-1] + 1]
                selected_nodes = np.where((self.computational_graph_degrees[layer] >= min_degree) & (self.computational_graph_degrees[layer] < max_degree))[0]
            else:
                min_degree = self.degree_hist_edges_log[layer][index[0]]
                max_degree = self.degree_hist_edges_log[layer][index[-1] + 1]
                log_degrees = np.log(self.computational_graph_degrees[layer] + 1)
                selected_nodes = np.where((log_degrees >= min_degree) & (log_degrees < max_degree))[0]
            pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
            self.pre_selected_nodes_stream.event(selected_nodes=np.union1d(pre_selected_nodes, selected_nodes))

    def get_fairness_metrics(self):
        return self.fairness_metric_view_value_bar.data[()]['Value']