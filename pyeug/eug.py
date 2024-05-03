from . import util
from . import draw
from . import community
from .css import scrollbar_css, multichoice_css
from . import RangesetCategorical
# from .metrics import node_classification
# from .metrics.pdd import pdd
from . import individual_bias
import numpy as np
import holoviews as hv
from holoviews import opts
import os
import pickle
import torch
import dgl
import numpy as np
from sklearn.manifold import TSNE
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from scipy import stats
from holoviews.streams import Stream, param
from holoviews import streams
# import datashader as ds
# import datashader.transfer_functions as tf
import pandas as pd
# from datashader.bundling import connect_edges, hammer_bundle
# from holoviews.operation.datashader import datashade, bundle_graph
import panel as pn
# from . import RangesetCategorical
import copy
from bokeh.models import HelpButton, Tooltip
import random
import sys
import time

sys.setrecursionlimit(1000000)

hv.extension('bokeh')
pn.extension()

pn.config.raw_css.append(scrollbar_css)
pn.config.raw_css.append(multichoice_css )

PERPLEXITY = 15 # german
SEED = 42

# set np seed
np.random.seed(SEED)


# class DataGraphView(Stream):
#     xy = param.Array(default=np.array([]), constant=False, doc='XY positions.')
#     sens_name = param.List(default=[], constant=True, doc='Sensitive attribute name.')


# class DataGraphGraphView(Stream):
#     # numpy array
#     source_id = param.Array(default=np.array([]), constant=True, doc='Source of edges.')
#     target_id = param.Array(default=np.array([]), constant=True, doc='Target of edges.')
#     sens = param.Array(default=np.array([]), constant=True, doc='Sensitive attributes.')
#     sampled_alpha = param.Array(default=np.array([]), constant=True, doc='Sampled alpha.')
#     sens_names = param.List(default=[], constant=True, doc='Sensitive attribute names.')
#     # bundle = param.Boolean(default=False, constant=True, doc='Whether to bundle edges.')


class DataDensityView(Stream):
    graph_metrics = param.List(default=[], constant=True, doc='Graph metrics.')


# class DataEmbeddingView(Stream):
#     sampled_alpha = param.Array(default=np.array([]), constant=True, doc='Sampled alpha.')
#     polygons_lst = param.List(default=[], constant=True, doc='Polygons.')
#     max_edges_lst = param.List(default=[], constant=True, doc='Max edges.')


class SampleIdx(Stream):
    sample_idx = param.Array(default=np.array([]), constant=False, doc='Sampled indices.')


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


class DataLegend(Stream):
    sens = param.Array(default=np.array([]), constant=True, doc='Sensitive attributes.')
    sens_names = param.List(default=[], constant=True, doc='Sensitive attribute names.')


class DataMetricView(Stream):
    metrics = param.List(default=[], constant=True, doc='Metrics.')


class DataCorrelationViewSelection(Stream):
    correlations = param.List(default=[], constant=False, doc='Correlations.')
    p_vals = param.List(default=[], constant=False, doc='P values.')


class SelectedNodes(Stream):
    selected_nodes = param.Array(default=np.array([], dtype=int), constant=False, doc='Selected nodes.')


class Groups(Stream):
    groups = param.Array(default=np.array([]), constant=False, doc='Groups.')


class Contributions(Stream):
    contributions = param.Array(default=np.array([]), constant=False, doc='Contributions.')
    contributions_selected_attrs = param.Array(default=np.array([0.]), constant=False, doc='Contributions selected attributes.')


class ContributionsSelectedNodes(Stream):
    contributions_selected_nodes = param.Number(default=0., constant=False, doc='Contributions selected nodes.')


class SelectedAttrsLs(Stream):
    selected_attrs_ls = param.List(default=[[]], constant=False, doc='Selected attributes list.')


class EUG:
    def __init__(self, model, adj, feat, sens, sens_names, masks, 
                 labels, emb_fnc, perplexity=PERPLEXITY, feat_names=None):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # if feat_names is None
        if feat_names is None:
            self.feat_names = np.array([str(i) for i in range(feat.shape[-1])])
        else:
            self.feat_names = feat_names
        # self.model = copy.deepcopy(model)        
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
        # self.max_hop = max_hop 
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

        # determine if each col is one-hot or not
        if len(feat.shape) > 2:
            feat_2d = feat.reshape(-1, feat.shape[-1])
        else:
            feat_2d = feat
        # self.feat_types = np.all(np.logical_or(feat_2d == 0, feat_2d == 1), axis=0)

        # self.sub_adj = None
        self.neighbors = None

        preprocessed = util.init(model, feat, g, emb_fnc)
        self.embeddings = preprocessed['embeddings']
        self.max_hop = len(self.embeddings)
        # self.degree_boxes = preprocessed['degree_boxes']
        # self.metrics = util.calc_metrics(self.embeddings, sens_names, sens, self.degree_boxes)
        # self.metric_types = list(self.metrics[0].columns)

        self.embeddings_pca = util.proj_emb(self.embeddings, 'pca')
        self.embeddings_tsne = []
        self.embeddings_umap = util.proj_emb(self.embeddings, 'umap')
        # print('umap is fine')

        self.xy = np.array(self.embeddings_umap)
        # self.masked_adj_unfair = np.zeros_like(adj)
        # self.masked_adj_fair = np.zeros_like(adj)
        # self.feat_mask_unfair = np.zeros(self.feat.shape[-1])
        # self.feat_mask_fair = np.zeros(self.feat.shape[-1])
        self.layers = list(range(1, len(self.embeddings)+1))

        self.colors = ['#F1B8B8', '#81A995', '#F1E9B8', '#907FA3', '#F1DAB8', 
                       '#7F89A1', '#D6E4AE', '#CB9AB0', '#738E96', '#F1E2B8', 
                       '#9A799E', '#B5D4A1', '#F1CEB8', '#8984A7', '#F1F1B8', ]
        
        # self.selectable_metrics = []
        # for layer in self.layers:
        #     for metric_type in self.metric_types:
        #         self.selectable_metrics.append(f'{layer}-{metric_type}')

        # sampled_alpha = 1000 1s 
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
        # self.feature_view = None
        self.correlation_view_selection_continuous = None

        self.records = []

        self.pre_compute_contours_cache = {}

        options_dict = {f'Hop-{i+1}': i for i in range(self.max_hop)}
        self.correlation_view_selection = pn.widgets.Select(name='Hops', options=options_dict,)

    def show(self):
        height = 900
        width = 1400
        self.height = height
        self.width = width
        padding = 15
        # self.control_panel_height = int(height/3-35-30)
        self.control_panel_height = int(height/3-35)-padding
        self.control_panel_width = int(width/4-20)
        self.graph_view_height = int(height/3-35)-padding
        self.graph_view_width = int(width/4)-padding
        # self.fairness_metric_view_height = int(height/3-35)-padding
        # self.fairness_metric_view_width = int(width/2)-padding
        self.fairness_metric_view_height = int(height*2/3-35)-padding
        self.fairness_metric_view_width = int(width/4)-padding
        self.node_selection_view_height = int(height/3-35)-padding
        self.node_selection_view_width = int(width*3/4)-padding
        self.density_view_height = int(height/3-35)-padding
        self.density_view_width = int(width/4)-padding
        self.structural_bias_overview_height = int(height*2/3-35)-padding
        self.structural_bias_overview_width = int(width/4)-padding
        # self.attribute_view_height = int(height/2/2-35)
        # self.attribute_view_width = int(width*10/14)
        self.correlation_view_height = int(height/3-35)-padding
        self.correlation_view_width = int(width/4)-padding
        self.diagnostic_panel_height = int(height*2/3-35)-padding
        self.diagnostic_panel_width = int(width*3/4)-padding

        # widget for all
        self.record_button = pn.widgets.Button(name='Record', button_type='default')
        self.record_button.on_click(self._record_button_callback)
        # self.sens_name_selector = pn.widgets.Select(options=self.sens_names, value = self.sens_names[0], name='Sensitive Attribute')
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
        # self.n_neighbors_scale_group.param.watch(self._n_neighbors_scale_group_callback, 'value')

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
        self.layer_selection = pn.widgets.Select(options=self.layers, name='Layer', value=self.layers[-1], width=200)

        self.projection_selection = pn.widgets.Select(options=['UMAP', 'PCA', 't-SNE'], name='Projection', width=200)
        # watch projection selection
        self.projection_selection.param.watch(self._projection_selection_callback, 'value')

        self.node_sample_size_slider = pn.widgets.IntSlider(name='Node Sample Size', 
                                                            start=0, 
                                                            end=self.n_nodes, 
                                                            step=1, 
                                                            # value=self.init_sample_size, 
                                                            value=300,
                                                            width=200)
        
        # create sampled_alpha
        node_sample_size = self.node_sample_size_slider.value
        self.sampled_alpha = np.zeros(self.n_nodes)
        sample_idx = np.random.choice(np.arange(self.n_nodes), node_sample_size, replace=False)
        sample_idx_stream = SampleIdx(sample_idx=sample_idx)
        self.sampled_alpha[sample_idx] = 1
        self.data_embedding_view_alpha = DataEmbeddingViewAlpha(alpha=self.sampled_alpha) 

        sample_button = pn.widgets.Button(name='Sample', button_type='default')
        sample_button.on_click(self._sample_button_callback)

        # sampled_alpha = np.zeros(self.n_nodes)
        # # self.init_sample_size = 200
        # self.init_sample_size = self.n_nodes
        # sample_idx = np.random.choice(np.arange(self.n_nodes), self.init_sample_size, replace=False)
        # sampled_alpha[sample_idx] = 1

        # self.data_embedding_view = DataEmbeddingView(sampled_alpha=sampled_alpha,
        #                                              xy=self.xy)
        self.data_embedding_view_xy = DataEmbeddingViewXy(xy=self.xy)
        self.data_embedding_view_xy.add_subscriber(self._prepare_data_embedding_view)
        self.groups_stream.add_subscriber(self._prepare_data_embedding_view)
        self.layer_selection.param.watch(self._prepare_data_embedding_view, 'value') 
        # self.data_embedding_view = DataEmbeddingView(sampled_alpha=sampled_alpha)
        self.data_embedding_view_polys = DataEmbeddingViewPolys()

        self.data_embedding_view_thr_range = DataEmbeddingViewThrRange()
        self.data_embedding_view_thr_range.add_subscriber(self._update_thr_range)

        self.embedding_view_thr_slider = pn.widgets.FloatSlider(name='RangeSet Threshold', 
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
        # print('reach polys')
        self.graph_view_polys = hv.DynamicMap(pn.bind(draw.draw_embedding_view_polys,
                                                      layer=self.layer_selection,
                                                      colors=self.colors,
                                                      threshold=self.embedding_view_thr_slider,),
            streams=[self.groups_stream,
                     self.data_embedding_view_xy,
                     self.data_embedding_view_polys,
                     ])
        # print('polys is fine')
        
        self.graph_view_square = hv.DynamicMap(pn.bind(draw.draw_embedding_view_square,
                                                       layer=self.layer_selection,),
            streams=[
                     self.data_embedding_view_xy,
                     self.graph_view_scatter_selection1d
                     ])
        
        self.graph_view_overlay = (self.graph_view_square * self.graph_view_scatter * self.graph_view_polys).opts(
            width=int(self.node_selection_view_width*0.26),
            height=int(self.node_selection_view_height*0.95),
            shared_axes=False,
        ) 
        
        self.graph_view_legend = hv.DynamicMap(pn.bind(draw.draw_embedding_view_legend,
                                                       colors=self.colors,),
            streams=[self.groups_stream]).opts(
                width=int(self.graph_view_width*0.17),
                height=int(self.graph_view_height*0.95),
            )

### fairness metric view
        self.fairness_metric_view_chart = pn.pane.HoloViews()
        self.fairness_metric_view_chart_column = pn.Column(self.fairness_metric_view_chart)

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
            height=int(self.fairness_metric_view_height*0.4),
        )
        self.fairness_metric_view_value_bar_selection1d = hv.streams.Selection1D(source=self.fairness_metric_view_value_bar)
        self.fairness_metric_view_value_bar_selection1d.add_subscriber(self._update_fairness_metric_detail)

        self.fairness_metric_view = pn.Card(
            pn.Column(self.fairness_metric_view_bar), 
            self.fairness_metric_view_chart_column,
            hide_header=True,
            name='Fairness Metric View',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.fairness_metric_view_height,
            width=self.fairness_metric_view_width,
        )

### correlation view
        feat_np = self.feat.cpu().numpy()

        # options_dict = {f'Hop-{i+1}': i for i in range(self.max_hop)}
        # self.correlation_view_selection = pn.widgets.Select(name='Hops', options=options_dict,
        #                                                     width=int(self.correlation_view_width*0.93),
        #     )

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
            # print('i:', i)
            # if i == 0:
            #     adj_mul = self.adj0
            # else:
            #     adj_mul = torch.sparse.mm(adj_mul, self.adj0)
            # d = adj_mul.sum(axis=1).to_dense()
            if i == 0:
                d = self.adj0.sum(axis=1).unsqueeze(1)
            else:
                d = torch.sparse.mm(self.adj0, d)
            computational_graph_degrees.append(d.to_dense().squeeze())
        # convert computational_graph_degrees to a np array
        self.computational_graph_degrees = torch.stack(computational_graph_degrees).cpu().numpy()  

        self.dependency_view_degree_sens = hv.DynamicMap(pn.bind(draw.draw_dependency_view_degree_sens,
                                                                 computational_graph_degrees=self.computational_graph_degrees,
                                                                 hop=self.correlation_view_selection,
                                                                 scale=self.n_neighbors_scale_group,),
                                                                 streams=[self.selected_nodes_stream,
                                                                          self.groups_stream,]).opts(
                                                                              height=int(self.diagnostic_panel_height*0.45),
                                                                              width=int(self.diagnostic_panel_width*0.3),
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
        # self.density_view_scatter = draw.draw_density_view_scatter(graph_metrics) \
        self.density_view_scatter = hv.DynamicMap(draw.draw_density_view_scatter,
                                                  streams=[self.data_density_view]).opts(
                  width=int(self.node_selection_view_width*0.3), 
                  height=int(self.node_selection_view_height*0.93)-60,
                  shared_axes=False,
                  )
        
        self.density_view_scatter_selection1d = hv.streams.Selection1D(source=self.density_view_scatter)
        # watch it
        self.density_view_scatter_selection1d.add_subscriber(self._update_selected_communities_dropdown)

        self.selected_communities_dropdown = pn.widgets.Select(name='Tapped Communities', options=[None],
                                                               width=int(self.correlation_view_width*0.93))
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

        self.bias_contributions_nodes = 1
        self.bias_contributions_attrs = contribution_attrs
        self.bias_contributions_structure = contribution_structure
        self.bias_contributions_emb = 1

        self.bias_contributions_nodes_latex = pn.pane.LaTeX('Nodes: 1')
        self.bias_contributions_attrs_latex = pn.pane.LaTeX(f'Attributes: {contribution_attrs:.4f}')
        self.bias_contributions_structure_latex = pn.pane.LaTeX(f'Structure: {contribution_structure:.4f}')
        self.bias_contributions_emb_latex = pn.pane.LaTeX(f'Embeddings: 1')

        self.attribute_view_overview.opts(
            height=int(self.diagnostic_panel_height*0.5),
            width=int(self.diagnostic_panel_width*0.83),
        )
        self.diagnostic_panel = pn.Card(
            pn.Row(
                pn.Column(
                    '#### Attribute Selection',
                    self.attr_selection_mode_button,
                    self.new_selection_button,
                    '#### Bias Contributions',
                    self.bias_contributions_nodes_latex,
                    self.bias_contributions_structure_latex,
                    self.bias_contributions_attrs_latex,
                    self.bias_contributions_emb_latex,
                ),
                self.attribute_view_overview,
            ),
            pn.Row(
                pn.Column(self.dependency_view_degree_sens),
                pn.Column(self.dependency_view_attr_sens),
                pn.Column(self.dependency_view_attr_degree),
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
            width=int(self.node_selection_view_width*0.3),
            height=int(self.node_selection_view_height*0.95),  
        )

        self.structural_bias_overview_hist_all_selection1d = hv.streams.Selection1D(source=self.structural_bias_overview_hist_all)
        self.structural_bias_overview_hist_all_selection1d.add_subscriber(self._structural_bias_overview_hist_all_selection1d_subscriber)

        self.node_selectin_view = pn.Card(
            pn.Row(
                self.graph_view_legend,   
                self.graph_view_overlay,
                pn.Column(
                    # self.correlation_view_selection,
                    self.structural_bias_overview_hist,
                ),
                pn.Column(
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
        self.control_panel = pn.Card(
            pn.Column(
                '### Global Settings', 
                self.record_button,
                '#### Sensitive Attribute Selection',
                pn.Row(
                    self.sens_name_selector, 
                    sens_name_confirm_control_panel_button,
                ), 
                '#### Node Selection',
                self.n_selected_nodes_latex,
                pn.Row(
                    node_selection_confirm_control_panel_button,
                    node_selection_clear_control_panel_button,
                ),
                '#### Scale of # of Neighbors',
                self.n_neighbors_scale_group,
                pn.layout.Divider(), 
                '### Node Selection View Settings',
                self.layer_selection, 
                self.projection_selection,
                pn.Row(
                    self.node_sample_size_slider,
                    sample_button,
                ),
                self.embedding_view_thr_slider,
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
        # edge_sample_size = self.edge_sample_size_slider.value
        if node_sample_size < self.n_nodes:
            sampled_alpha = np.zeros(self.n_nodes)
            sample_idx = np.random.choice(np.arange(self.n_nodes), node_sample_size, replace=False)
            # self.sampled_idx_stream.event(sampled_idx=sample_idx)
            sampled_alpha[sample_idx] = 1
            self.sampled_alpha = sampled_alpha
        else:
            self.sampled_alpha = np.ones(self.n_nodes)
        
        selected_nodes = self.pre_selected_nodes_stream.selected_nodes
        if len(selected_nodes) > 0:
            # create a np array containing 1 and 0.2 representing the alpha value of the nodes selected and not selected
            selected_alpha = np.zeros(self.n_nodes)
            selected_alpha[selected_nodes] = 1
            # not selected nodes are set to 0.2
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
                                height=int(self.diagnostic_panel_height*0.45),
                                width=int(self.diagnostic_panel_width*0.3),
                                ) 
                            
                            self.dependency_view_attr_degree.object = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_degree_violin,
                                                                                            feat=feat,
                                                                                            computational_graph_degrees=self.computational_graph_degrees,
                                                                                            hop=self.correlation_view_selection,
                                                                                            scale=self.n_neighbors_scale_group,),
                                                                                    streams=[self.selected_nodes_stream,
                                                                                    ]
                                                                                    ).opts(
                                                                                        height=int(self.diagnostic_panel_height*0.45),
                                                                                        width=int(self.diagnostic_panel_width*0.3),
                                                                                    )
                        else:
                            self.dependency_view_attr_sens.object = hv.DynamicMap(pn.bind(draw.draw_dependency_view_attr_sens_violin,
                                                                                   feat=feat),
                                                                           streams=[self.selected_nodes_stream,
                                                                                    self.groups_stream]).opts(
                                                                                        height=int(self.diagnostic_panel_height*0.45),
                                                                                        width=int(self.diagnostic_panel_width*0.3),
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
                                height=int(self.diagnostic_panel_height*0.45),
                                width=int(self.diagnostic_panel_width*0.3),
                                shared_axes=False
                            )
                else:
                    self.dependency_view_attr_sens.object = None
                    self.dependency_view_attr_degree.object = None
                    if not id.empty:
                        id = id.values[0]
                        if id in self.selected_attrs_ls_stream.selected_attrs_ls[-1]:
                            # self.selected_attrs_ls_stream.selected_attrs_ls[-1].remove(id)
                            selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
                            selected_attrs_ls[-1].remove(id)
                            self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
                        else:
                            # self.selected_attrs_ls_stream.selected_attrs_ls[-1].append(id)
                            selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
                            selected_attrs_ls[-1].append(id)
                            self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
                
                # calculate the deleting area
                selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
                circle_gap = 2
                r_sector1 = 0.45
                n_selected_attrs = len(selected_attrs_ls)
                n_all_nodes = self.n_nodes
                n = self.n_feat
                hw_ratio = int(self.diagnostic_panel_height*0.5)/int(self.diagnostic_panel_width*0.93)
                ymax = n+1.5
                ymin = 0 
                xmax = n_all_nodes * 1.1 
                xmin = (circle_gap * xmax * (n_selected_attrs + 1)) / (circle_gap * n_selected_attrs + circle_gap - hw_ratio * ymax + hw_ratio * ymin)
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
                            # self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
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
            # selected_nodes = self.node_indices
            selected_nodes = np.array([])
        # self.selected_nodes_stream.event(selected_nodes=selected_nodes)
        pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
        self.pre_selected_nodes_stream.event(selected_nodes=np.union1d(pre_selected_nodes, selected_nodes))

    def _get_data_mask(self):
        """
        Create a mask for the selected data categories (train, val, test, other).
        """
        mask = torch.zeros_like(self.train_mask).bool()
        if 'Train' in self.data_selector.value:
            mask = mask | self.train_mask
        if 'Val' in self.data_selector.value:
            mask = mask | self.val_mask
        if 'Test' in self.data_selector.value:
            mask = mask | self.test_mask
        if 'Unlabeled' in self.data_selector.value:
            mask = mask | self.other_mask 
        return mask

    # def _sens_name_confirm_control_panel_button_callback(self, event):
    #     sens_name = self.sens_name_selector.value
    #     self.graph_view_scatter.event(sens_name=sens_name)
    #     self.graph_view_xdist.event(sens_name=sens_name)
    #     self.graph_view_ydist.event(sens_name=sens_name)
    #     # self._confirm_control_panel_button_callback(event)
    #     self._update_fairness_metric_view()

    # def _update_fairness_metric_detail(self, event, metric_name):
    def _update_fairness_metric_detail(self, index):
        # print('index:', index)
        if index:
            metric_name = index[0]
            chart = hv.DynamicMap(pn.bind(draw.draw_fairness_metric_view_detail,
                                        metric_name=metric_name,
                                        #   metric_name_dict=self.metric_name_dict,
                                        labels=self.labels,
                                        predictions=self.predictions,
                                        eod_radio=self.fairness_metric_view_chart_eod_radio),
                                streams=[self.selected_nodes_stream, self.groups_stream]) 

            if metric_name == 2 or metric_name == 6:
                if len(self.fairness_metric_view_chart_column) == 1:
                    self.fairness_metric_view_chart_column.insert(0, self.fairness_metric_view_chart_eod_radio)
                    chart.opts(width=int(self.fairness_metric_view_width*0.93), height=int(self.fairness_metric_view_height*0.45)) 
            else:  
                if len(self.fairness_metric_view_chart_column) == 2: 
                    self.fairness_metric_view_chart_column.pop(0)   
                chart.opts(width=int(self.fairness_metric_view_width*0.93), height=int(self.fairness_metric_view_height*0.55)) 

            self.fairness_metric_view_chart.object = chart
        else:
            self.fairness_metric_view_chart.object = None


    def _min_threshold_slider_button_callback(self, event):
        thr = self.min_threshold_slider.value
        # Recalculate communities
        # self.node_communities = community.extract_communities(community.Tree(), 
        #                                                       *copy.deepcopy(self.extract_communities_args[1: 3]), 
        #                                                       self.extract_communities_args[3], 
        #                                                       thr)
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
        # self.density_view_scatter.object = draw.draw_density_view_scatter(graph_metrics)
        self.data_density_view.event(graph_metrics=graph_metrics)

    def _pre_selected_nodes_stream_subscriber(self, selected_nodes):
        if len(selected_nodes) > 0:
            selected_alpha = np.zeros(self.n_nodes)
            # selected_alpha[selected_nodes] = 1
            selected_alpha[self.pre_selected_nodes_stream.selected_nodes.astype(int)] = 1
            # selected_alpha[selected_alpha == 0] = 0 
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

        # # update self.contributions_selected_nodes_stream
        # gc = individual_bias.group_bias_contribution(
        #     self.adj,
        #     self.feat,
        #     self.model,
        #     self.groups_stream.groups,
        #     self.selected_nodes_stream.selected_nodes
        # )
        # self.contributions_selected_nodes_stream.event(contributions_selected_nodes=gc)

        # update self.bias_contributions_attrs_latex
        contribution_nodes = individual_bias.group_bias_contribution(
            adj=self.adj,
            model=self.model,
            features=self.feat,
            group=self.groups_stream.groups,
            selected_nodes=self.selected_nodes_stream.selected_nodes
        )
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
        contribution_emb = individual_bias.calc_emb_contribution(
            model=self.model,
            g=self.g, 
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups,
        )
        self.bias_contributions_nodes = contribution_nodes
        self.bias_contributions_attrs = contribution_attrs
        self.bias_contributions_structure = contribution_structure
        self.bias_contributions_emb = contribution_emb
        self.bias_contributions_nodes_latex.object = f'Nodes: {contribution_nodes:.4f}'
        self.bias_contributions_attrs_latex.object = f'Attributes: {contribution_attrs:.4f}'
        self.bias_contributions_structure_latex.object = f'Structure: {contribution_structure:.4f}'
        self.bias_contributions_emb_latex.object = f'Embeddings: {contribution_emb:.4f}'

    # def _contributions_selected_nodes_stream_subscriber(self, contributions_selected_nodes):
    #     # update correlation_view_md
    #     # self.correlation_view_latex2.object = f'{(contributions_selected_nodes * 100):.2f}%'
    #     self.bias_contributions_nodes_latex.object = f'Nodes: {contributions_selected_nodes:.4f}'

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
            # self.data_embedding_view.event(xy=np.array(self.embeddings_umap))
            self.data_embedding_view_xy.event(xy=np.array(self.embeddings_umap))
        elif projection_name == 't-SNE':
            # check if self.embeddings_tsne is an empty list
            if not self.embeddings_tsne:
                # compute the embeddings
                self.embeddings_tsne = util.proj_emb(self.embeddings, 'tsne')
            # update the embeddings
            # self.data_embedding_view.event(xy=np.array(self.embeddings_tsne))
            self.data_embedding_view_xy.event(xy=np.array(self.embeddings_tsne))
        elif projection_name == 'PCA':
            # check if self.embeddings_pca is an empty list
            if not self.embeddings_pca:
                # compute the embeddings
                self.embeddings_pca = util.proj_emb(self.embeddings, 'pca')
            # update the embeddings
            # self.data_embedding_view.event(xy=np.array(self.embeddings_pca))
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
        # print('pre contours is fine')
        # update polygons_lst, max_edges_lst in the data_embedding_view
        # self.data_embedding_view.event(polygons_lst=polygons_lst, max_edges_lst=max_edges_lst)
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
            # self.contributions_stream.contributions_selected_attrs[-1] = contribution_selected_attrs
            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
            contributions_selected_attrs[-1] = contribution_selected_attrs
        else:
            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
            contributions_selected_attrs[-1] = 0.

        self.contributions_stream.event(contributions_selected_attrs=contributions_selected_attrs) 

    def _save_extract_communities_args(self):
        # Determine the directory of the current file
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'extract_communities_args.pkl')

        # Use pickle to serialize self.extract_communities_args to the file
        with open(file_path, 'wb') as file:
            # pickle.dump(self.extract_communities_args, file) 
            pickle.dump(self.extract_communities_args[1: 3], file)
            # pickle.dump(self.extract_communities_args[1], file)

    def _load_extract_communities_args(self):
        # Determine the directory of the current file
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'extract_communities_args.pkl')

        # Use pickle to deserialize the object from the file
        with open(file_path, 'rb') as file:
            # self.extract_communities_args = pickle.load(file)
            ret = pickle.load(file)
        return ret

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
        selected_nodes = self.selected_nodes_stream.selected_nodes
        selected_attrs = self.selected_attrs_ls_stream.selected_attrs_ls[-1] 
        self.records.append({
            'Sens': self.sens_name_selector.value,
            'Nodes': selected_nodes, 
            'Attributes': selected_attrs,
            'Bias Contribution of Nodes': self.bias_contributions_nodes,
            'Bias Contribution of Attributes': self.bias_contributions_attrs,
            'Bias Contribution of Structure': self.bias_contributions_structure,
            'Bias Contribution of Embeddings': self.bias_contributions_emb,
        })   

    def _graph_view_scatter_selection1d_subscriber(self, index):
        # device of self.adj0
        device = self.adj0.device
        if index:
            if len(index) > 1:
                # self.selected_nodes_stream.event(selected_nodes=np.array(index))
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
                # adj_mul_indices_current = self.adj_mul_indices[self.layer_selection.value - 1]
                # selected_nodes = adj_mul_indices_current[1, adj_mul_indices_current[0] == idx].cpu().numpy()
                # if idx not in selected_nodes:
                #     selected_nodes = np.append(selected_nodes, idx)

                # self.selected_nodes_stream.event(selected_nodes=selected_nodes)
                pre_selected_nodes = self.pre_selected_nodes_stream.selected_nodes
                self.pre_selected_nodes_stream.event(selected_nodes=np.union1d(pre_selected_nodes, selected_nodes))
                
        else:
            # self.selected_nodes_stream.event(selected_nodes=self.node_indices)
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
        
        # options = self.selected_communities_dropdown.options
        # if None not in options:
        #     options.insert(0, None)
        #     self.selected_communities_dropdown.options = options
        self.selected_communities_dropdown.value = None
        print('options:', self.selected_communities_dropdown.options)

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