from . import util
from . import draw
from . import community
from .css import scrollbard_css
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

hv.extension('bokeh')
pn.extension()

pn.config.raw_css.append(scrollbard_css)

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


# class DataFeatureView(Stream):
#     feat = param.Array(default=np.array([]), constant=True, doc='Features.')
#     hop_indicator_computation_graph = param.List(default=[], constant=True, doc='Computation graph.')
#     hop_indicator_induced_unfair = param.List(default=[], constant=True, doc='Unfair induced hop indicator.')
#     hop_indicator_induced_fair = param.List(default=[], constant=True, doc='Fair induced hop indicator.')
#     # max_hop = param.Integer(default=0, constant=True, doc='Max hop.')


class DataSubgraphView(Stream):
    communities = param.List(default=[], constant=True, doc='Communities.')


class DataCorrelationViewSelection(Stream):
    correlations = param.List(default=[], constant=False, doc='Correlations.')
    p_vals = param.List(default=[], constant=False, doc='P values.')


class SelectedNodes(Stream):
    selected_nodes = param.Array(default=np.array([]), constant=False, doc='Selected nodes.')


class Groups(Stream):
    groups = param.Array(default=np.array([]), constant=False, doc='Groups.')


class Contributions(Stream):
    contributions = param.Array(default=np.array([]), constant=False, doc='Contributions.')
    contributions_selected_attrs = param.Array(default=np.array([0.]), constant=False, doc='Contributions selected attributes.')


class ContributionsSelectedNodes(Stream):
    contributions_selected_nodes = param.List(default=[], constant=False, doc='Contributions selected nodes.')


class SelectedAttrsLs(Stream):
    selected_attrs_ls = param.List(default=[[]], constant=False, doc='Selected attributes list.')


class IndividualBiasMetrics(Stream):
    individual_bias_metrics = param.Array(default=np.array([]), constant=False, doc='Individual bias metrics.')


class EUG:
    def __init__(self, model, adj, feat, sens, sens_names, max_hop, masks, 
                 labels, emb_fnc, perplexity=PERPLEXITY, feat_names=None):
        # if feat_names is None
        if feat_names is None:
            self.feat_names = np.array([str(i) for i in range(feat.shape[-1])])
        else:
            self.feat_names = feat_names
        # self.model = copy.deepcopy(model)        
        self.model = model
        self.model.eval()
        self.adj = adj
        self.adj_mul_indices = []
        self.adj0, self.adj1, self.adj0_scipy, self.adj1_scipy = util.modify_sparse_tensor_scipy(adj)
        self.feat = feat
        self.n_feat = feat.shape[-1]
        self.sens = sens
        self.groups_stream = Groups(groups=self.sens[0])
        self.sens_names = sens_names
        self.max_hop = max_hop
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
        # self.degree_boxes = preprocessed['degree_boxes']
        # self.metrics = util.calc_metrics(self.embeddings, sens_names, sens, self.degree_boxes)
        # self.metric_types = list(self.metrics[0].columns)

        self.embeddings_pca = util.proj_emb(self.embeddings, 'pca')
        self.embeddings_tsne = []
        self.embeddings_umap = util.proj_emb(self.embeddings, 'umap')
        print('umap is fine')

        self.xy = np.array(self.embeddings_umap)
        # self.masked_adj_unfair = np.zeros_like(adj)
        # self.masked_adj_fair = np.zeros_like(adj)
        # self.feat_mask_unfair = np.zeros(self.feat.shape[-1])
        # self.feat_mask_fair = np.zeros(self.feat.shape[-1])
        self.layers = list(range(1, len(self.embeddings)+1))

        # self.metric_name_dict = {
        #                 # 'SD MF1': 'micro_f1_std_dev',
        #                 # 'DI': 'surrogate_di',
        #                 # 'EFPR': 'efpr',
        #                 # 'EFNR': 'efnr',
        #                 # 'ETPR': 'etpr',
        #                 'D SP': 'delta_dp_metric',
        #                 'Dm SP': 'max_diff_delta_dp',
        #                 'D EO': 'delta_eo_metric',
        #                 'D Acc': 'delta_accuracy_metric',
        #                 'SD Acc': 'sigma_accuracy_metric',
        #                 # 'D TED': 'delta_ted_metric',
        #                 }

        # self.colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', 
        #     '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', 
        #     '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', 
        #     '#4e7670']
        # self.colors = [
        #     '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', 
        #     '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', 
        #     '#4e7670']
        # self.colors = ['#D17474',  '#D19874', '#D1AB74',  '#D1B874', 
        #     '#D1C474', '#D1D174', '#AEC66E',  '#86B766', '#519271', 
        #     '#4A7682', '#54658C', '#615890', '#6E538D', '#834D89', 
        #     '#AF6285'] 
        # self.colors = [
        #     '#AF6285',
        #     '#D19874',
        #     '#519271',
        #     '#AEC66E',
        #     '#6E538D',
        #     '#4A7682', 
        #     '#D1C474',
        #     '#834D89', 
        #     '#D1AB74',
        #     '#D1D174',
        #     '#615890',
        #     '#86B766',
        #     '#D17474',
        #     '#54658C', 
        #     '#D1B874']
        # self.colors = ['#E7F9B5',
        # '#FFD5B9',
        # '#FFFFB9',
        # '#FFB9B9',
        # '#FFECB9',
        # '#A4B4DA',
        # '#FFF5B9',
        # '#D39ED8',
        # '#BDA3DA',
        # '#EDACCA',
        # '#B0A8DC',
        # '#9CC8D4',
        # '#C9F1AF',
        # '#FFE3B9', 
        # '#A1DDBF']
        self.colors = ['#F1B8B8', '#F1CEB8', '#F1DAB8', '#F1E2B8', '#F1E9B8', '#F1F1B8', 
                       '#D6E4AE', '#B5D4A1', '#81A995', '#738E96', '#7F89A1', '#8984A7', 
                       '#907FA3', '#9A799E', '#CB9AB0',]
        
        # self.selectable_metrics = []
        # for layer in self.layers:
        #     for metric_type in self.metric_types:
        #         self.selectable_metrics.append(f'{layer}-{metric_type}')

        # sampled_alpha = 1000 1s 
        self.n_nodes = adj.shape[0]
        self.node_indices = np.arange(self.n_nodes)
        # self.selected_nodes = self.node_indices 
        # self.previous_selected_nodes = self.selected_nodes
        # self.selected_nodes_stream = SelectedNodes(selected_nodes=self.selected_nodes)
        self.selected_nodes_stream = SelectedNodes(selected_nodes=self.node_indices)
        self.selected_nodes_stream.add_subscriber(self._selected_nodes_stream_subscriber)
        # sampled_alpha = np.zeros(self.n_nodes)
        # # self.init_sample_size = 200
        # self.init_sample_size = self.n_nodes
        # sample_idx = np.random.choice(np.arange(self.n_nodes), self.init_sample_size, replace=False)
        # sampled_alpha[sample_idx] = 1
        # convert adj to pd.dataframe consisting of two cols, source and target representing the edges
        # adj_triu = np.triu(adj, k=1)
        # edges = adj_triu.nonzero()
        # self.edges = np.array(edges).T
        # indices_adj = adj.coalesce().indices()
        # self.edges = indices_adj[: , indices_adj[0] > indices_adj[1]].cpu().numpy().T
        # sampled_edges = self.edges[np.isin(self.edges[:,0], sample_idx)]
        # sampled_edges = sampled_edges[np.isin(sampled_edges[:,1], sample_idx)]
        # np.random.shuffle(sampled_edges)
        # self.init_edge_sample_size = 200
        # if sampled_edges.shape[0] > self.init_edge_sample_size:
        #     sampled_edges = sampled_edges[: self.init_edge_sample_size]
        # source = sampled_edges[:, 0]
        # target = sampled_edges[:, 1]

        # # self.data_graph_view = DataGraphView(xy=self.xy)
        # self.data_graph_graph_view = DataGraphGraphView(source_id=source, target_id=target, 
        #     sens=self.sens, sens_names=self.sens_names, sampled_alpha=sampled_alpha)
        # self.data_legend = DataLegend(sens=self.sens, sens_names=self.sens_names)
        # self.data_metric_view = DataMetricView(metrics=self.metrics)
        # self.data_feature_view = DataFeatureView(feat=self.feat)

        self.previous_attribute_view_overview_tap_x = None
        self.previous_attribute_view_overview_tap_y = None

        # self.selection = streams.Selection1D()
        # self.ids = self.selection.index        
        # self.threshold_slider = pn.widgets.FloatSlider(name='Threshold', start=0, end=1.0, step=0.001, value=0.999)
        self.graph_view_scatter = None
        self.graph_view = None
        self.metric_view = None
        # self.feature_view = None
        self.correlation_view_selection_continuous = None

        self.records = []

    def show(self):
        height = 900
        width = 1400
        self.height = height
        self.width = width
        padding = 15
        self.control_panel_height = int(height/3-35-30)
        self.control_panel_width = int(width/4-20)
        self.graph_view_height = int(height/3-35)-padding
        self.graph_view_width = int(width/4)-padding
        self.fairness_metric_view_height = int(height/3-35)-padding
        self.fairness_metric_view_width = int(width/2)-padding
        self.density_view_height = int(height/3-35)-padding
        self.density_view_width = int(width/4)-padding
        self.subgraph_view_height = int(height/3-35)-padding
        self.subgraph_view_width = int(width*3/4/2)-padding
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
        # self.sens_name_confirm_control_panel_button = pn.widgets.Button(name='Confirm', button_type='default')
        # self.sens_name_confirm_control_panel_button.on_click(self._sens_name_confirm_control_panel_button_callback)

        # widget for metric view
        # selected_metrics = pn.widgets.MultiChoice(name='Metrics', value=[],
        #     options=self.selectable_metrics)

        # widget for fairness metric view
        # a multi-select widget for selecting "train", "val", "test", "other"
        self.data_selector = pn.widgets.MultiChoice(name='Data', value=['Train', 'Val', 'Test', 'Unlabeled'],
            options=['Train', 'Val', 'Test', 'Unlabeled'],
            width=int(self.control_panel_width*0.63),
            )
        self.previous_data_selector_value = self.data_selector.value
        sens_name_confirm_control_panel_button = pn.widgets.Button(name='Confirm', button_type='default')
        sens_name_confirm_control_panel_button.on_click(self._sens_name_confirm_control_panel_button_callback)
        data_select_control_panel_button = pn.widgets.Button(name='Select', button_type='default')
        data_select_control_panel_button.on_click(self._data_select_control_panel_button_callback)

### graph view
        # wh_graph_view = min(width/3, height/2-115)
        # wh_graph_view = min(self.graph_view_width, self.graph_view_height-35)
        # self.data_graph_view = DataGraphView(xy=self.xy, sens_name=self.sens_name_selector.value)

        # self.graph_view_scatter = hv.DynamicMap(pn.bind(draw.draw_graph_view, layer=self.layer_selection, bundle=bundle, colors=self.colors), 
        #     # streams=[self.data_graph_graph_view, self.data_graph_view, self.selection]) \
        #     streams=[self.data_graph_graph_view, self.data_graph_view]) \
        #     .opts(framewise=True, 
        #           xaxis=None, yaxis=None, 
        #         #   width=int(wh_graph_view*0.75), 
        #         #   height=int(wh_graph_view*0.75),
        #             width=int(self.graph_view_width*0.93),
        #             height=int(self.graph_view_height*0.95),
        #           shared_axes=False,
        #           ) 

        # widget for graph view
        self.layer_selection = pn.widgets.Select(options=self.layers, name='Layer', width=200)

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
        ##### move self.individual_bias_metrics_stream here ####
        node_sample_size = self.node_sample_size_slider.value
        self.sampled_alpha = np.ones(self.n_nodes)
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
        # self.graph_view_scatter_selection1d = hv.streams.Selection1D()
        # self.graph_view_scatter = hv.DynamicMap(pn.bind(draw.draw_embedding_view, 
        #                                                 layer=self.layer_selection, 
        #                                                 colors=self.colors,
        #                                                 threshold=self.embedding_view_thr_slider,), 
        #     streams=[self.groups_stream, 
        #              self.data_embedding_view_xy,
        #              self.data_embedding_view,
        #              self.graph_view_scatter_selection1d,
        #              ]) \
        #     .opts(
        #           xaxis=None, yaxis=None, 
        #             width=int(self.graph_view_width*0.6),
        #             height=int(self.graph_view_height*0.95),
        #           shared_axes=False,
        #           )
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
        print('reach polys')
        self.graph_view_polys = hv.DynamicMap(pn.bind(draw.draw_embedding_view_polys,
                                                      layer=self.layer_selection,
                                                      colors=self.colors,
                                                      threshold=self.embedding_view_thr_slider,),
            streams=[self.groups_stream,
                     self.data_embedding_view_xy,
                     self.data_embedding_view_polys,
                     ])
        print('polys is fine')
        
        self.graph_view_square = hv.DynamicMap(pn.bind(draw.draw_embedding_view_square,
                                                       layer=self.layer_selection,),
            streams=[
                     self.data_embedding_view_xy,
                     self.graph_view_scatter_selection1d
                     ])
        
        self.graph_view_overlay = (self.graph_view_square * self.graph_view_scatter * self.graph_view_polys).opts(
            width=int(self.graph_view_width*0.7),
            height=int(self.graph_view_height*0.95),
            shared_axes=False,
        ) 
        
        self.graph_view_legend = hv.DynamicMap(pn.bind(draw.draw_embedding_view_legend,
                                                       colors=self.colors,),
            streams=[self.groups_stream]).opts(
                width=int(self.graph_view_width*0.2),
                height=int(self.graph_view_height*0.95),
            )
        # self.graph_view_legend = pn.pane.HoloViews()
        # self.graph_view_legend.object = draw.draw_embedding_view_legend(colors=self.colors,
        #                                                                 groups=self.groups_stream.groups)

        # self.graph_view_xdist = hv.DynamicMap(pn.bind(draw.draw_distribution, layer=self.layer_selection, sens=self.sens, sens_names=self.sens_names, 
        #     colors=self.colors, x_or_y='x'),
        #     streams=[self.data_graph_view]) \
        #     .opts(height=int(wh_graph_view*0.25))

        # self.graph_view_ydist = hv.DynamicMap(pn.bind(draw.draw_distribution, layer=self.layer_selection, sens=self.sens, sens_names=self.sens_names,
        #     colors=self.colors, x_or_y='y'),
        #     streams=[self.data_graph_view]) \
        #     .opts(width=int(wh_graph_view*0.25))

        # scatter = (self.graph_view_scatter + self.graph_view_legend).opts(
        #     hooks=[lambda plot, element: setattr(plot.state.toolbar, 'logo', None)],
        # )
        
        self.graph_view = pn.Card(
            # self.graph_view_scatter << self.graph_view_ydist << self.graph_view_xdist,
            # self.graph_view_scatter + self.graph_view_legend,
            pn.Row(
                # self.graph_view_scatter,
                self.graph_view_legend,   
                self.graph_view_overlay,
            ),
            # (self.graph_view_overlay + self.graph_view_legend), 
            # scatter,
            hide_header=True,
            name='Embedding View',
            # height=int(height/2-80), 
            # width=int(width/3),
            height=self.graph_view_height,
            width=self.graph_view_width,
            )
        
        self.graph_view_widgets = pn.WidgetBox(
            self.layer_selection, 
            self.projection_selection,
            self.node_sample_size_slider,
            sample_button,
            # width=int(width/3), 
            # height=int(height/2-80),
            height=self.graph_view_height,
            width=self.graph_view_width,
            name='Settings',
            )

        # legend = hv.DynamicMap(pn.bind(draw.draw_legend, sens_name=self.sens_name_selector, colors=self.colors), streams=[self.data_legend])

        # edge_legend = draw.draw_edge_legend()

### fairness metric view
        self.fairness_metric_view_chart = pn.pane.HoloViews()
        self.fairness_metric_view_chart_column = pn.Column(self.fairness_metric_view_chart)

        self.fairness_metric_view_chart_eod_radio = pn.widgets.RadioButtonGroup(
            options=['TPR', 'FPR'], value='TPR', button_type='default', width=200)

        # self.fairness_metric_view_bar_tap = streams.SingleTap()
        # self.fairness_metric_view_bar = hv.DynamicMap(pn.bind(draw.draw_fairness_metric_view_bar, 
        #                                                       labels=self.labels,
        #                                                       predictions=self.predictions),
        #                                               streams=[self.selected_nodes_stream, 
        #                                                        self.groups_stream,
        #                                                     #    self.fairness_metric_view_bar_tap
        #                                                        ])
        # self.fairness_metric_view_bar_selection1d = hv.streams.Selection1D(source=self.fairness_metric_view_bar)
        # self.fairness_metric_view_bar.opts(
        #     opts.Bars(width=int(self.fairness_metric_view_width*0.5),
        #               height=int(self.fairness_metric_view_height*0.95)),
        # )
        # self.fairness_metric_view_bar_tap.add_subscriber(self._update_fairness_metric_detail)

        self.fairness_metric_view_value_bar = hv.DynamicMap(pn.bind(draw.draw_fairness_metric_view_value_bar,
                                                                     labels=self.labels,
                                                                     predictions=self.predictions),
                                                                streams=[self.selected_nodes_stream, 
                                                                        self.groups_stream,
                                                                        ])
        self.fairness_metric_view_range_bar = draw.draw_fairness_metric_view_range_bar()
        self.fairness_metric_view_bar = (self.fairness_metric_view_value_bar * self.fairness_metric_view_range_bar).opts(
            width=int(self.fairness_metric_view_width*0.5),
            height=int(self.fairness_metric_view_height*0.95),
        )
        self.fairness_metric_view_value_bar_selection1d = hv.streams.Selection1D(source=self.fairness_metric_view_value_bar)
        self.fairness_metric_view_value_bar_selection1d.add_subscriber(self._update_fairness_metric_detail)

        self.fairness_metric_view = pn.Card(
            pn.Row(
                # pn.Column(*self.items), 
                pn.Column(self.fairness_metric_view_bar), 
                self.fairness_metric_view_chart_column,
                ),
            hide_header=True,
            name='Fairness Metric View',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.fairness_metric_view_height,
            width=self.fairness_metric_view_width,
        )

### correlation view
        print('reach correlation view')
        # idx_train = self.train_mask.nonzero().flatten()
        # idx_test = self.test_mask.nonzero().flatten()
        # idx_val = self.val_mask.nonzero().flatten()
        # estimated_pdd = pdd(self.adj, 
        #                     idx_train, 
        #                     self.feat, 
        #                     idx_test, 
        #                     torch.from_numpy(self.labels), 
        #                     self.sens[0], 
        #                     idx_val, 
        #                     self.model, 
        #                     self.g)
        self.contributions_selected_nodes_stream = ContributionsSelectedNodes(contributions_selected_nodes=[])
        self.contributions_selected_nodes_stream.add_subscriber(self._contributions_selected_nodes_stream_subscriber)
        print('reach individual bias')
        self.individual_bias_metrics_stream = IndividualBiasMetrics(
            individual_bias_metrics=individual_bias.individual_bias_contribution(self.adj, self.feat, self.model, self.sens[0],
                                                                                 ))
        print('individual bias is fine')
        # self.individual_bias_metrics = individual_bias.individual_bias_contribution(self.adj, self.feat, self.model, self.sens[0])
        # feat_train = self.feat[self.train_mask].cpu().numpy()
        feat_np = self.feat.cpu().numpy()
        # self.columns_categorical = util.check_unique_values(feat_train)
        self.columns_categorical = util.check_unique_values(feat_np)
        # # pearson_r_results = [stats.pearsonr(row, estimated_pdd) for row in feat_train.T[~self.columns_categorical]]
        # pearson_r_results = [stats.pearsonr(row, self.individual_bias_metrics) 
        #                      for row in feat_np.T[~self.columns_categorical]]
        # pearson_correlations = [r[0] for r in pearson_r_results]
        # pearson_p_vals = [r[1] for r in pearson_r_results]

        # # pointbiserialr_results = [stats.pointbiserialr(row, estimated_pdd) for row in feat_train.T[self.columns_categorical]]
        # pointbiserialr_results = [stats.pointbiserialr(row, self.individual_bias_metrics) 
        #                           for row in feat_np.T[self.columns_categorical]]
        # pointbiserial_correlations = [r[0] for r in pointbiserialr_results]
        # pointbiserial_p_vals = [r[1] for r in pointbiserialr_results]

        # degree corr
        computational_graph_degrees = []
        for i in range(self.max_hop):
            if i == 0:
                adj_mul = self.adj0
            else:
                adj_mul = torch.sparse.mm(adj_mul, self.adj0)
            d = adj_mul.sum(axis=1).to_dense()
            computational_graph_degrees.append(d)

            indices = adj_mul.coalesce().indices()
            self.adj_mul_indices.append(indices)
        print('degree is fine')

        # convert computational_graph_degrees to a np array
        self.computational_graph_degrees = torch.stack(computational_graph_degrees).cpu().numpy()    
        # pearson_r_results_degree = [stats.pearsonr(row, self.individual_bias_metrics) 
        pearson_r_results_degree = [stats.pearsonr(row, self.individual_bias_metrics_stream.individual_bias_metrics) 
                                    for row in self.computational_graph_degrees]
        pearson_correlations_degree = [r[0] for r in pearson_r_results_degree]
        pearson_p_vals_degree = [r[1] for r in pearson_r_results_degree]  

        # self.data_correlation_view_selection_continuous = DataCorrelationViewSelection(correlations=pearson_correlations, p_vals=pearson_p_vals)
        # self.data_correlation_view_selection_categorical = DataCorrelationViewSelection(correlations=pointbiserial_correlations, p_vals=pointbiserial_p_vals)
        # self.data_correlation_view_selection_degree = DataCorrelationViewSelection(correlations=pearson_correlations_degree, p_vals=pearson_p_vals_degree)
        
        # self.height_correlation_view_selection_continuous = 40
        # self.height_correlation_view_selection_categorical = self.height_correlation_view_selection_continuous
        # self.height_correlation_view_selection_degree = self.height_correlation_view_selection_continuous
        # self.height_correlation_view_selection_degree = 40
        # self.width_correlation_view_selection_continuous = self.height_correlation_view_selection_continuous * len(pearson_correlations)
        # self.width_correlation_view_selection_categorical = self.height_correlation_view_selection_continuous * len(pointbiserial_correlations)
        # self.width_correlation_view_selection_degree = self.height_correlation_view_selection_degree * len(pearson_correlations_degree)

        # # self.correlation_view_selection_continuous = draw.draw_correlation_view_selection(pearson_correlations, pearson_p_vals, self.feat_names[~self.columns_categorical]) \
        # #     .opts(opts.Rectangles(height=self.height_correlation_view_selection_continuous, width=self.width_correlation_view_selection_continuous, ),).opts(shared_axes=False,)
        # single_tap_correlation_view_selection_continuous = streams.SingleTap(x=0, y=0)
        # self.correlation_view_selection_continuous = hv.DynamicMap(pn.bind(draw.draw_correlation_view_selection, text=self.feat_names[~self.columns_categorical],),
        #                                                            streams=[self.data_correlation_view_selection_continuous, single_tap_correlation_view_selection_continuous]) \
        #     .opts(opts.Rectangles(height=self.height_correlation_view_selection_continuous, width=self.width_correlation_view_selection_continuous, ),).opts(shared_axes=False,)
        # # self.correlation_view_hex = hv.DynamicMap(pn.bind(draw.draw_correlation_view_hex, feat=feat_train.T[~self.columns_categorical], metric=estimated_pdd),
        # self.correlation_view_hex = hv.DynamicMap(pn.bind(draw.draw_correlation_view_hex, feat=feat_np.T[~self.columns_categorical], metric=self.individual_bias_metrics),
        #                                         #   streams=[streams.SingleTap(source=self.correlation_view_selection_continuous)]) \
        #                                             streams=[single_tap_correlation_view_selection_continuous, self.selected_nodes_stream]) \
        #     .opts(width=int(self.correlation_view_width*0.4), 
        #           height=int(self.correlation_view_height*0.6),
        #           shared_axes=False,
        #           )
        
        # # self.correlation_view_selection_categorical = draw.draw_correlation_view_selection(pointbiserial_correlations, pointbiserial_p_vals, self.feat_names[self.columns_categorical]) \
        # #     .opts(opts.Rectangles(height=self.height_correlation_view_selection_categorical, width=self.width_correlation_view_selection_categorical, ),).opts(shared_axes=False,)
        # single_tap_correlation_view_selection_categorical = streams.SingleTap(x=0, y=0)
        # self.correlation_view_selection_categorical = hv.DynamicMap(pn.bind(draw.draw_correlation_view_selection, text=self.feat_names[self.columns_categorical],),
        #                                                              streams=[self.data_correlation_view_selection_categorical, single_tap_correlation_view_selection_categorical]) \
        #     .opts(opts.Rectangles(height=self.height_correlation_view_selection_categorical, width=self.width_correlation_view_selection_categorical, ),).opts(shared_axes=False,)
        # # self.correlation_view_violin = hv.DynamicMap(pn.bind(draw.draw_correlation_view_violin, feat=feat_train.T[self.columns_categorical], metric=estimated_pdd),
        # self.correlation_view_violin = hv.DynamicMap(pn.bind(draw.draw_correlation_view_violin, feat=feat_np.T[self.columns_categorical], metric=self.individual_bias_metrics),
        #                                             # streams=[streams.SingleTap(source=self.correlation_view_selection_categorical)]) \
        #                                             streams=[single_tap_correlation_view_selection_categorical, self.selected_nodes_stream]) \
        #     .opts(width=int(self.correlation_view_width*0.4),
        #             height=int(self.correlation_view_height*0.6),
        #             shared_axes=False,
        #             )
        
        # self.correlation_view_selection_degree = draw.draw_correlation_view_selection(pearson_correlations_degree, pearson_p_vals_degree, [f'{i+1}' for i in range(self.max_hop)]) \
        #     .opts(opts.Rectangles(
        #         height=self.height_correlation_view_selection_degree, 
        #         width=self.width_correlation_view_selection_degree,
        #         ),).opts(shared_axes=False,)
        # single_tap_correlation_view_selection_degree = streams.SingleTap(x=0, y=0)
        # self.correlation_view_selection_degree = hv.DynamicMap(pn.bind(draw.draw_correlation_view_selection, text=[f'{i+1}' for i in range(self.max_hop)],),
        #                                                         streams=[self.data_correlation_view_selection_degree, single_tap_correlation_view_selection_degree]) \
        #     .opts(opts.Rectangles(
        #         height=self.height_correlation_view_selection_degree, 
        #         width=self.width_correlation_view_selection_degree,
        #         ),).opts(shared_axes=False,)
        # options_dict = {f'Hop-{i+1} Corr.: {pearson_correlations_degree[i]:.2f} P-value: {pearson_p_vals_degree[i]:.2f}': i for i in range(self.max_hop)}
        options_dict = {f'Hop-{i+1} Corr.: {pearson_correlations_degree[i]:.2f}': i for i in range(self.max_hop)}
        self.correlation_view_selection = pn.widgets.Select(name='Hops', options=options_dict,
                                                            width=int(self.correlation_view_width*0.93),
            )
        
        # self.correlation_view_degree = hv.DynamicMap(pn.bind(draw.draw_correlation_view_hex, feat=self.computational_graph_degrees, metric=self.individual_bias_metrics),
        #                                             # streams=[streams.SingleTap(source=self.correlation_view_selection_degree)]) \
        #                                             streams=[single_tap_correlation_view_selection_degree, self.selected_nodes_stream]) \
        # self.correlation_view_degree = hv.DynamicMap(pn.bind(draw.draw_correlation_view_hex, 
        #                                                      feat=self.computational_graph_degrees, 
        #                                                     #  metric=self.individual_bias_metrics,
        #                                                      index=self.correlation_view_selection),
        #                                             # streams=[streams.SingleTap(source=self.correlation_view_selection_degree)]) \
        #                                             streams=[self.selected_nodes_stream, 
        #                                                      self.individual_bias_metrics_stream,]) \
        #     .opts(width=int(self.correlation_view_width*0.93),
        #             height=int(self.correlation_view_height*0.95)-60,
        #             shared_axes=False,
        #             )

        # self.correlation_view_rectangles = hv.DynamicMap(draw.draw_correlation_view_rectangles,
        #                                                  streams=[self.contributions_selected_nodes_stream]).opts(
        #     width=int(self.correlation_view_width*0.93),
        #     height=int(self.correlation_view_height*0.1),
        #     shared_axes=False,
        # )

        self.correlation_view_latex1 = pn.pane.LaTeX('Bias Contributions')

        self.correlation_view_latex2 = pn.pane.LaTeX('Separate: 0.00%    Composite: 0.00%')   
 
        self.correlation_view_hex = hv.DynamicMap(pn.bind(draw.draw_correlation_view_hex, 
                                                             feat=self.computational_graph_degrees, 
                                                            #  metric=self.individual_bias_metrics,
                                                             index=self.correlation_view_selection),
                                                    # streams=[streams.SingleTap(source=self.correlation_view_selection_degree)]) \
                                                    streams=[self.individual_bias_metrics_stream,])
        self.correlation_view_scatter = hv.DynamicMap(pn.bind(draw.draw_correlation_view_scatter,
                                                                feat=self.computational_graph_degrees, 
                                                                index=self.correlation_view_selection,
                                                                colors=self.colors,),
                                                        streams=[self.groups_stream, 
                                                                 self.individual_bias_metrics_stream,
                                                                 self.data_embedding_view_alpha,])
        
        self.correlation_view_scatter_selection1d = hv.streams.Selection1D(source=self.correlation_view_scatter)
        # self.correlation_view_scatter_selection1d.add_subscriber(self._correlation_view_scatter_selection1d_subscriber)
        self.correlation_view_scatter_selection1d.add_subscriber(self._graph_view_scatter_selection1d_subscriber)

        self.correlation_view_scatter_square = hv.DynamicMap(pn.bind(draw.draw_correlation_view_square,
                                                                     feat=self.computational_graph_degrees,
                                                                     hop=self.correlation_view_selection,),
            streams=[self.individual_bias_metrics_stream,
                     self.correlation_view_scatter_selection1d,
                    ])


        self.correlation_view_degree = (self.correlation_view_hex * self.correlation_view_scatter_square * self.correlation_view_scatter).opts(width=int(self.correlation_view_width*0.93),
            height=int(self.correlation_view_height*0.95)-60,
            shared_axes=False, 
            framewise=True
        )

        self.correlation_view = pn.Card(
            # pn.Row(
            #     # pn.Column(
            #     #     self.correlation_view_selection_continuous, 
            #     #     self.correlation_view_selection_categorical,
            #     #     sizing_mode='stretch_height',
            #     # ),
            #     self.correlation_view_selection_continuous, 
            #     # sizing_mode='stretch_height', 
            #     scroll=True, 
            #     width=self.correlation_view_width
            # ),
            # pn.Row(
            #     self.correlation_view_selection_categorical, 
            #     sizing_mode='stretch_height', 
            #     scroll=True, 
            #     width=self.correlation_view_width
            # ),
            # pn.Row(
            #     self.correlation_view_selection_degree,
            #     sizing_mode='stretch_height',
            #     scroll=False,
            #     width=self.correlation_view_width
            # ),
            # correlation_view_legend, 
            # pn.Row(
                # self.correlation_view_hex,
                # self.correlation_view_violin,
            self.correlation_view_selection,
            self.correlation_view_degree,
                # single_tap_correlation_view_selection_continuous,
                # single_tap_correlation_view_selection_categorical,
                # single_tap_correlation_view_selection_degree,
            # ),
            hide_header=True,
            name='Correlation View',
            height=self.correlation_view_height,
            width=self.correlation_view_width,
            )
        print('correlation view is fine')

### attribute view
        # # continuous attributes
        # feat_continuous = self.feat[:, ~self.columns_categorical].cpu().numpy()
        # feat_names_continuous = self.feat_names[~self.columns_categorical]
        # feat_continuous_df = pd.DataFrame(feat_continuous, columns=feat_names_continuous)
        # # Initialize an empty list to hold the plots
        # self.attribute_view_violins = []
        # # Iterate over each feature name in the DataFrame
        # for feat_name in feat_names_continuous:
        #     # Extract the column (variable) data
        #     variable_data = feat_continuous_df[feat_name]          
        #     # Generate the violin plot for this variable
        #     plot = draw.draw_attribute_view_violin(variable_data, feat_name, self.sens[0], self.selected_nodes)   
        #     # plot = hv.DynamicMap(pn.bind(draw.draw_attribute_view_violin, variable_data=variable_data, feat_name=feat_name), 
        #     #                      streams=[self.selected_nodes_stream, self.groups_stream])
        #     # Append the generated plot to the list of plots
        #     self.attribute_view_violins.append(plot)
        # # Combine all plots vertically into a single layout
        # self.attribute_view_violin = pn.pane.HoloViews(hv.Layout(self.attribute_view_violins).cols(1))

        # # categorical attributes
        # feat_categorical = self.feat[:, self.columns_categorical].cpu().numpy()
        # feat_names_categorical = self.feat_names[self.columns_categorical]
        # feat_categorical_df = pd.DataFrame(feat_categorical, columns=feat_names_categorical)
        # # Initialize an empty list to hold the bar charts
        # self.attribute_view_bars = []
        # # Iterate over each feature name in the DataFrame
        # for feat_name in feat_names_categorical:
        #     # Extract the column (variable) data
        #     variable_data = feat_categorical_df[feat_name]            
        #     # Generate the bar chart for this variable
        #     # bar_chart = draw.draw_attribute_view_bar(variable_data, feat_name, self.sens[0])
        #     bar_chart = hv.DynamicMap(pn.bind(draw.draw_attribute_view_bar, variable_data=variable_data, feat_name=feat_name),
        #                               streams=[self.selected_nodes_stream, self.groups_stream])            
        #     # Append the generated chart to the list
        #     self.attribute_view_bars.append(bar_chart)
        # # Combine all bar charts vertically into a single layout
        # self.attribute_view_bar = hv.Layout(self.attribute_view_bars).cols(1)

        # # sensitive attribute
        # # self.attribute_view_sens = draw.draw_attribute_view_sens(self.sens[0])
        # self.attribute_view_sens = hv.DynamicMap(draw.draw_attribute_view_sens, 
        #                                          streams=[self.selected_nodes_stream, self.groups_stream])

        # overview
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
        self.attribute_view_overview = hv.DynamicMap(pn.bind(draw.draw_attribute_view_overview, 
                                                             feat=pd.DataFrame(self.feat.cpu().numpy(), columns=self.feat_names),
                                                             columns_categorical=self.columns_categorical,
                                                             hw_ratio=int(self.diagnostic_panel_height*0.5)/int(self.diagnostic_panel_width*0.93)
                                                            #  individual_bias_metrics=self.individual_bias_metrics,
                                                             ),
                                                        streams=[self.selected_nodes_stream, 
                                                                 self.groups_stream,
                                                                 self.attribute_view_overview_tap,
                                                                 self.individual_bias_metrics_stream,
                                                                 self.contributions_stream,
                                                                 self.selected_attrs_ls_stream,])
        # watch the tap
        self.attribute_view_overview_tap.add_subscriber(self._attribute_view_overview_tap_subscriber)
        # detail
        self.attribute_view_detail = pn.pane.HoloViews()
        # check if the 0-th feat is continuous or not
        # if self.columns_categorical[0]:
        #     self.attribute_view_detail.object = draw.draw_attribute_view_bar(self.feat[:, 0].cpu().numpy(), self.feat_names[0], self.sens[0], self.selected_nodes)
        # else:
        #     self.attribute_view_detail.object = draw.draw_attribute_view_violin(self.feat[:, 0].cpu().numpy(), self.feat_names[0], self.sens[0], self.selected_nodes)

        # correlation
        self.attribute_view_correlation = pn.pane.HoloViews()
        # check if the 0-th feat is continuous or not
        
        # self.attribute_view = pn.Card(
        #     # hv.Layout([self.attribute_view_sens, ]).cols(1),
        #     # pn.Row(
        #     #     pn.Column(
        #     #         self.attribute_view_violin, 
        #     #         scroll=True, 
        #     #         height=int(self.attribute_view_height*0.6)
        #     #     ),
        #     #     pn.Column(
        #     #         self.attribute_view_bar, 
        #     #         scroll=True, 
        #     #         height=int(self.attribute_view_height*0.6)
        #     #     ),  
        #     # ),
        #     self.attribute_view_overview,
        #     # self.attribute_view_overview_tap,
        #     self.attribute_view_detail,
        #     self.attribute_view_correlation,
        #     hide_header=True,
        #     name='Attribute View',
        #     height=self.attribute_view_height,
        #     width=self.attribute_view_width,
        #     )
        print('attribute view is fine')

### density view
        # Create the scatter plot
        print('reach process graph')
        self.extract_communities_args = community.process_graph(self.adj0)
        print('process graph is fine')
        self._save_extract_communities_args()
        loaded_extract_communities_args = self._load_extract_communities_args()
        # deep copy self.extract_communities_args, which is a tuple
        self.node_communities = community.extract_communities(community.Tree(), 
                                                              *loaded_extract_communities_args, 
                                                              self.extract_communities_args[3], 
                                                              0.5)
        print('communities is fine')

        # self.node_communities = community.extract_communities(community.Tree(), 
        #                                                       *copy.deepcopy(self.extract_communities_args[1: 3]), 
        #                                                       self.extract_communities_args[3], 
        #                                                       0.5)
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
        # self.data_subgraph_view = DataSubgraphView(communities=self.communities)
        graph_metrics = util.calculate_graph_metrics(self.communities) # Adjust this line as per your actual data structure
        self.data_density_view = DataDensityView(graph_metrics=graph_metrics)
        # self.density_view_scatter = draw.draw_density_view_scatter(graph_metrics) \
        self.density_view_scatter = hv.DynamicMap(draw.draw_density_view_scatter,
                                                  streams=[self.data_density_view]).opts(
                  width=int(self.density_view_width*0.93), 
                  height=int(self.density_view_height*0.95)-60,
                  shared_axes=False,
                  )
        
        self.density_view_scatter_selection1d = hv.streams.Selection1D(source=self.density_view_scatter)
        # watch it
        self.density_view_scatter_selection1d.add_subscriber(self._update_selected_communities_dropdown)

        self.selected_communities_dropdown = pn.widgets.Select(name='Selected Nodes', options=[None],
                                                               width=int(self.correlation_view_width*0.93))
        # watch it
        self.selected_communities_dropdown.param.watch(self._update_subgraph_view, 'value')
        # Slider for min_threshold
        self.min_threshold_slider = pn.widgets.FloatSlider(name='Minimum Density Threshold', start=0.0, end=1.0, step=0.01, value=0.5, width=200)

        # Button to recalculate and redraw plots
        min_threshold_slider_button = pn.widgets.Button(name='Update Communities')

        min_threshold_slider_button.on_click(self._min_threshold_slider_button_callback)

        self.density_view_widgets = pn.WidgetBox(
            self.min_threshold_slider, 
            min_threshold_slider_button,
            height=self.density_view_height,
            width=self.density_view_width,
            name='Settings',
            )

        self.density_view = pn.Card(
            self.selected_communities_dropdown,
            self.density_view_scatter,
            hide_header=True,
            name='Density View',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.density_view_height,
            width=self.density_view_width,
            )
        
### subgraph view
        # Create the heatmap (initially with index 0)
        # self.subgraph_view_heatmap = hv.DynamicMap(draw.draw_subgraph_view_heatmap, 
        #                                           streams=[hv.streams.Selection1D(source=self.density_view_scatter), 
        #                                                    self.data_subgraph_view]) \
        self.subgraph_view_heatmap = hv.DynamicMap(pn.bind(draw.draw_subgraph_view_heatmap, adj=self.adj0_scipy), 
                                                  streams=[self.selected_nodes_stream]).opts(
            height=int(self.diagnostic_panel_height*0.45),
            width=int(self.diagnostic_panel_height*0.45),
        )
            # .opts(width=int(self.subgraph_view_width), 
            #       height=int(self.subgraph_view_height),
            #       shared_axes=False,
            #       )
        
        # self.subgraph_view = pn.Card(
        #     self.subgraph_view_heatmap,
        #     hide_header=True,
        #     name='Subgraph View',
        #     # height=int(height/2-80),
        #     # width=int(width/3),
        #     height=self.subgraph_view_height,
        #     width=self.subgraph_view_width,
        #     )
        
### diagnostic panel
        self.attr_selection_mode_button = pn.widgets.RadioButtonGroup(
            name='Selection Mode',
            options=['Single', 'Multiple'],
            value='Single',
        )

        self.new_selection_button = pn.widgets.Button(name='New Selection')
        self.new_selection_button.on_click(self._new_selection_button_callback)

        self.attribute_view_overview.opts(
            height=int(self.diagnostic_panel_height*0.5),
            width=int(self.diagnostic_panel_width*0.83),
        )
        self.diagnostic_panel = pn.Card(
            # pn.Row(
            #     self.attribute_view_overview,
            #     self.attribute_view_detail,
            # ),
            # pn.Row(
            #     self.attribute_view_correlation,
            #     self.subgraph_view,
            # ),
            # pn.Row(
            #     pn.Column(
            #         self.attribute_view_overview,
            #         self.attribute_view_detail,
            #     ),
            #     pn.Column(
            #         self.subgraph_view_heatmap,
            #         self.attribute_view_correlation,
            #     ),
            # ),
            pn.Row(
                pn.Column(
                    self.attr_selection_mode_button,
                    self.new_selection_button
                ),
                self.attribute_view_overview,
            ),
            pn.Row(
                pn.Column(self.subgraph_view_heatmap),
                pn.Column(self.attribute_view_detail),
                pn.Column(self.attribute_view_correlation),
            ),
            hide_header=True,
            name='Diagnostic Panel',
            height=self.diagnostic_panel_height,
            width=self.diagnostic_panel_width,
        )

### structural bias overview
        self.structural_bias_overview = pn.Card(
            self.correlation_view_latex1,
            self.correlation_view_latex2, 
            self.correlation_view_selection,
            # self.correlation_view_rectangles,
            self.correlation_view_degree,
            self.selected_communities_dropdown,
            self.density_view_scatter,
            hide_header=True,
            name='Structural Bias Overview',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.structural_bias_overview_height,
            width=self.structural_bias_overview_width, 
        )       
# control panel
        self.control_panel = pn.Card(
            pn.Column(
                '### Global Control',
                pn.Row(
                    self.sens_name_selector, 
                    sens_name_confirm_control_panel_button,
                ),
                pn.Row(
                    self.data_selector,
                    data_select_control_panel_button,
                ),
                self.record_button,
                pn.layout.Divider(), 
                '### Embedding View Settings',
                self.layer_selection, 
                self.projection_selection,
                self.node_sample_size_slider,
                sample_button,
                self.embedding_view_thr_slider,
                pn.layout.Divider(),
                '### Structural Bias Overview Settings',
                self.min_threshold_slider, 
                min_threshold_slider_button,
                scroll=True,
            ),
            name='Control Panel',
            hide_header=True, 
            height=self.control_panel_height, 
            width=self.control_panel_width,
        )
        
        # app = pn.GridSpec(width=self.width, height=self.height)
        # app[0: 2, 0: 4] = pn.Tabs(
        #     pn.WidgetBox(
        #         self.sens_name_selector, 
        #         sens_name_confirm_control_panel_button,
        #         self.data_selector,
        #         data_select_control_panel_button,
        #         name='Control Panel',
        #         # height=int(height/2-80), width=int(width/3),
        #         height=self.control_panel_height, 
        #         width=self.control_panel_width,
        #     ),
        #     # height=int(height/2), 
        #     # width=int(width/3),
        #     # height=self.control_panel_height+80,
        #     # width=self.control_panel_width,
        #     )
        # app[0: 2, 4: 8] = pn.Tabs(
        #     self.graph_view,
        #     self.graph_view_widgets, 
        #     # height=int(height/2), 
        #     # width=int(width/3),
        #     # height=self.graph_view_height+80,
        #     # width=self.graph_view_width,
        #     )
        # app[0: 2, 8: 14] = pn.Tabs(
        #     self.fairness_metric_view,
        #     # self.fairness_metric_view_widgets,
        #     # height=int(height/2), 
        #     # width=int(width/3),
        #     # height=self.fairness_metric_view_height+80,
        #     # width=self.fairness_metric_view_width,
        #     )
        # app[2: 5, 10: 14] = pn.Tabs(
        #     self.density_view,
        #     self.density_view_widgets,
        #     )
        # app[6: 9, 10: 14] = pn.Tabs(
        #     self.subgraph_view,
        #     )
        # app[6: 11, 0: 10] = pn.Tabs(
        #     self.attribute_view,
        #     )
        # app[2: 6, 0: 10] = pn.Tabs(
        #     self.correlation_view,
        #     # height=self.density_view_height+80,
        #     # width=self.density_view_width,
        #     )

        app = pn.GridSpec(width=self.width, height=self.height)
        app[0, 0] = pn.Tabs(
            self.control_panel,
        )
        # app[1, 0] = pn.Tabs(
        #     self.correlation_view,
        # )
        # app[2, 0] = pn.Tabs(
        #     self.density_view,
        #     self.density_view_widgets,
        app[1: 3, 0] = pn.Tabs(
            self.structural_bias_overview,
        )
        app[0, 1: 3] = pn.Tabs(
            self.fairness_metric_view,
        )
        app[0, 3] = pn.Tabs(
            self.graph_view,
            # self.graph_view_widgets,
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
            self.sampled_idx_stream.event(sampled_idx=sample_idx)
            sampled_alpha[sample_idx] = 1
            self.sampled_alpha = sampled_alpha
        else:
            self.sampled_alpha = np.ones(self.n_nodes)
        # # convert adj to pd.dataframe consisting of two cols, source and target representing the edges
        # sampled_edges = self.edges[np.isin(self.edges[:,0], sample_idx)]
        # sampled_edges = sampled_edges[np.isin(sampled_edges[:,1], sample_idx)]
        # np.random.shuffle(sampled_edges)
        # if sampled_edges.shape[0] > edge_sample_size:
        #     sampled_edges = sampled_edges[: edge_sample_size]
        # source = sampled_edges[:, 0]
        # target = sampled_edges[:, 1]

        # self.graph_view_scatter.event(sampled_alpha=sampled_alpha, 
        #                             #   source_id=source, 
        #                             #   target_id=target
        #                               )
        
        selected_nodes = self.selected_nodes_stream.selected_nodes
        # create a np array containing 1 and 0.2 representing the alpha value of the nodes selected and not selected
        selected_alpha = np.zeros(self.n_nodes)
        selected_alpha[selected_nodes] = 1
        # not selected nodes are set to 0.2
        selected_alpha[selected_alpha == 0] = 0
        alpha = selected_alpha * self.sampled_alpha 

        self.data_embedding_view_alpha.event(alpha=alpha)

    def _sens_name_confirm_control_panel_button_callback(self, event):
        # # check if the data_selector value has changed
        # if self.data_selector.value != self.previous_data_selector_value:
        #     # update self.selected_nodes
        #     mask = self._get_data_mask()
        #     self.selected_nodes = self.node_indices[mask]
        # if not np.array_equal(self.selected_nodes, self.previous_selected_nodes): # notice the order
        #     self.selected_nodes_stream.event(selected_nodes=self.selected_nodes)
        if self.sens_name_selector.value != self.previous_sens_name_value:
            # Find the indices of the selected sensitive attributes
            sens_indices = [self.sens_names.index(name) for name in self.sens_name_selector.value]
            sens_selected = self.sens[sens_indices]
            vectorized_concat = np.vectorize(util.concatenate_elements)
            # Apply the vectorized function across columns
            sens = vectorized_concat(*sens_selected)
            self.groups_stream.event(groups=sens)

        # update fairness metric view
        # self._update_fairness_metric_view()

        # update graph view
        # self._update_graph_view(self.previous_selected_nodes)

        # update correlation view
        self._update_correlation_view()

        # update attribute view
        # self._update_attribute_view()

        # update subgraph view
        # self._update_subgraph_view()

        # self.previous_data_selector_value = self.data_selector.value
        # self.previous_selected_nodes = self.selected_nodes
        self.previous_sens_name_value = self.sens_name_selector.value

    # def _data_select_control_panel_button_callback(self, event):
    #     previous_selected_nodes = self.previous_selected_nodes
    #     # # check if the data_selector value has changed
    #     # if self.data_selector.value != self.previous_data_selector_value:
    #     # # update self.selected_nodes
    #     mask = self._get_data_mask()
    #     self.selected_nodes = self.node_indices[mask]
    #     # set the value of self.density_view_scatter_selection1d to None
    #     self.density_view_scatter_selection1d.event(index=[])
    #     if not np.array_equal(self.selected_nodes, self.previous_selected_nodes): # notice the order
    #         mask = self._get_data_mask()
    #         self.selected_nodes = self.node_indices[mask]
    #         self.selected_nodes_stream.event(selected_nodes=self.selected_nodes)

    #     # if not np.array_equal(self.selected_nodes, previous_selected_nodes):
    #     #     # update fairness metric view
    #     #     self._update_fairness_metric_view()

    #     # update graph view
    #     # self._update_graph_view(previous_selected_nodes)

    #     if not np.array_equal(self.selected_nodes, previous_selected_nodes):
    #         # update correlation view
    #         self._update_correlation_view()

    #     # update attribute view
    #     # self._update_attribute_view()

    #     # update subgraph view
    #     # self._update_subgraph_view()

    #     self.previous_data_selector_value = self.data_selector.value
    #     self.previous_selected_nodes = self.selected_nodes
        
    def _data_select_control_panel_button_callback(self, event):
        # # update self.selected_nodes
        mask = self._get_data_mask()
        selected_nodes = self.node_indices[mask]
        # set the value of self.density_view_scatter_selection1d to None
        self.density_view_scatter_selection1d.event(index=[])
        self.selected_nodes_stream.event(selected_nodes=selected_nodes)

        self.previous_data_selector_value = self.data_selector.value

    def _update_fairness_metric_view(self):
        pass
        # """
        # Update the fairness metrics based on the given sensitive attribute.
        # """
        # # # Find the indices of the selected sensitive attributes
        # # sens_indices = [self.sens_names.index(name) for name in self.sens_name_selector.value]
        # # sens_selected = self.sens[sens_indices]
        # # vectorized_concat = np.vectorize(util.concatenate_elements)
        # # # Apply the vectorized function across columns
        # # sens = vectorized_concat(*sens_selected)
        # sens = self.groups_stream.groups

        # # mask = self._get_data_mask()
        # mask = self.selected_nodes
        # predictions = self.predictions[mask]
        # labels = self.labels[mask]
        # sens = sens[mask]
        # labeled_mask = labels != -1
        # predictions = predictions[labeled_mask]
        # labels = labels[labeled_mask]
        # sens = sens[labeled_mask]

        # # Recalculate metrics
        # # You can add more metrics as per your requirement
        # metrics = {
        #     'SD MF1': node_classification.micro_f1_std_dev(predictions, labels, sens),
        #     'EFPR': node_classification.efpr(predictions, labels, sens),
        #     'EFNR': node_classification.efnr(predictions, labels, sens),
        #     'D SP': node_classification.delta_dp_metric(predictions, labels, sens),
        #     'Dm SP': node_classification.max_diff_delta_dp(predictions, labels, sens),
        #     'D EO': node_classification.delta_eo_metric(predictions, labels, sens),
        #     'D Acc': node_classification.delta_accuracy_metric(predictions, labels, sens),
        #     'SD Acc': node_classification.sigma_accuracy_metric(predictions, labels, sens),
        # }

        # # Update the labels with new metric values
        # for metric_widget, metric_name in zip(self.metric_widgets, metrics.keys()):
        #     metric_widget.value = metrics[metric_name]
        # # return metrics
            
    def _update_graph_view(self, previous_selected_nodes):
        pass
        # # if self.selected_nodes != self.previous_selected_nodes:
        # if not np.array_equal(self.selected_nodes, previous_selected_nodes):
        #     # create a np array containing 1 and 0.2 representing the alpha value of the nodes selected and not selected
        #     selected_alpha = np.zeros(self.n_nodes)
        #     selected_alpha[self.selected_nodes] = 1
        #     # not selected nodes are set to 0.2
        #     selected_alpha[selected_alpha == 0] = 0.2
        #     # element-wise multiplication of the selected_alpha and the sampled_alpha
        #     # alpha = selected_alpha * self.data_graph_graph_view.sampled_alpha
        #     alpha = selected_alpha * self.data_embedding_view.sampled_alpha
        #     # update the alpha of the nodes
        #     self.graph_view_scatter.event(sampled_alpha=alpha)
        # if self.sens_name_selector.value != self.previous_sens_name_value:
        #     self.graph_view_scatter.event(sens_name=self.sens_name_selector.value)
            # self.graph_view_xdist.event(sens_name=self.sens_name_selector.value)
            # self.graph_view_ydist.event(sens_name=self.sens_name_selector.value)
            # self.graph_view_legend.opts(
            #     ylim=(-0.3, 0.3*len(np.unique(self.groups_stream.groups))),
            # )

    def _update_correlation_view(self):
        if self.sens_name_selector.value != self.previous_sens_name_value:
            self.individual_bias_metrics_stream.event(individual_bias_metrics=individual_bias.individual_bias_contribution(self.adj, self.feat, self.model, self.groups_stream.groups))
        selected_individual_bias_metrics = self.individual_bias_metrics_stream.individual_bias_metrics[self.selected_nodes_stream.selected_nodes]
        # feat_np_selected = self.feat[self.selected_nodes].cpu().numpy()
        computational_graph_degrees_selected = self.computational_graph_degrees[:, self.selected_nodes_stream.selected_nodes]

        # pearson_r_results = [stats.pearsonr(row, selected_individual_bias_metrics) 
        #                      for row in feat_np_selected.T[~self.columns_categorical]]
        # pearson_correlations = [r[0] for r in pearson_r_results]
        # pearson_p_vals = [r[1] for r in pearson_r_results]

        # pointbiserialr_results = [stats.pointbiserialr(row, selected_individual_bias_metrics)
        #                           for row in feat_np_selected.T[self.columns_categorical]]
        # pointbiserial_correlations = [r[0] for r in pointbiserialr_results]
        # pointbiserial_p_vals = [r[1] for r in pointbiserialr_results]

        if len(selected_individual_bias_metrics) > 1:
            pearson_r_results_degree = [stats.pearsonr(row, selected_individual_bias_metrics) 
                                        for row in computational_graph_degrees_selected]
            pearson_correlations_degree = [r[0] for r in pearson_r_results_degree]
            pearson_p_vals_degree = [r[1] for r in pearson_r_results_degree]
        else:
            # set values to np.nan
            pearson_correlations_degree = [np.nan for _ in range(self.max_hop)]
            pearson_p_vals_degree = [np.nan for _ in range(self.max_hop)]

        # update the options of self.correlation_view_selection
        # options_dict = {f'Hop-{i+1} Corr.: {pearson_correlations_degree[i]:.2f} P-value: {pearson_p_vals_degree[i]:.2f}': i for i in range(self.max_hop)}
        options_dict = {f'Hop-{i+1} Corr.: {pearson_correlations_degree[i]:.2f}': i for i in range(self.max_hop)}
        self.correlation_view_selection.options = options_dict
        # print('done')
        # self.data_correlation_view_selection_continuous.event(correlations=pearson_correlations, p_vals=pearson_p_vals)
        # self.data_correlation_view_selection_categorical.event(correlations=pointbiserial_correlations, p_vals=pointbiserial_p_vals)
        # self.data_correlation_view_selection_degree.event(correlations=pearson_correlations_degree, p_vals=pearson_p_vals_degree)

    # def _update_attribute_view(self):
    #     # continuous attributes
    #     feat_continuous = self.feat[:, ~self.columns_categorical].cpu().numpy()
    #     feat_names_continuous = self.feat_names[~self.columns_categorical]
    #     feat_continuous_df = pd.DataFrame(feat_continuous, columns=feat_names_continuous)
    #     # Initialize an empty list to hold the plots
    #     self.attribute_view_violins = []
    #     # Iterate over each feature name in the DataFrame
    #     for feat_name in feat_names_continuous:
    #         # Extract the column (variable) data
    #         variable_data = feat_continuous_df[feat_name]          
    #         # Generate the violin plot for this variable
    #         plot = draw.draw_attribute_view_violin(variable_data, feat_name, self.groups_stream.groups, self.selected_nodes)   
    #         # plot = hv.DynamicMap(pn.bind(draw.draw_attribute_view_violin, variable_data=variable_data, feat_name=feat_name), 
    #         #                      streams=[self.selected_nodes_stream, self.groups_stream])
    #         # Append the generated plot to the list of plots
    #         self.attribute_view_violins.append(plot)
    #     self.attribute_view_violin.object = hv.Layout(self.attribute_view_violins).cols(1)
    def _attribute_view_overview_tap_subscriber(self, x, y):
        x, y = y, x
        if not (self.previous_attribute_view_overview_tap_x == x and self.previous_attribute_view_overview_tap_y == y):
            if y:
                # get the data of self.attribute_view_overview
                # print(self.attribute_view_overview.data)
                df = self.attribute_view_overview.data[()].get('Rectangles.I').data
                # find the corresponding id satisfying the y0 <= y <= y1
                id = df[(df['y0'] <= y) & (df['y1'] >= y)]['ID']
                if self.attr_selection_mode_button.value == 'Single':
                    if id.empty:
                        self.attribute_view_detail.object = None
                        self.attribute_view_correlation.object = None
                    else:
                        id = id.values[0]
                        feat = self.feat[:, id].cpu().numpy()
                        feat_name = self.feat_names[id]
                        if self.columns_categorical[id]:
                            self.attribute_view_detail.object = hv.DynamicMap(pn.bind(draw.draw_attribute_view_bar, 
                                                                                    variable_data=feat,
                                                                                    feat_name=feat_name),
                                                                            streams=[self.selected_nodes_stream, self.groups_stream]).opts(
                                                                                                                                            height=int(self.diagnostic_panel_height*0.45),
                                                                                                                                            width=int(self.diagnostic_panel_width*0.4),
                                                                                                                                        )
                            self.attribute_view_correlation.object = hv.DynamicMap(pn.bind(draw.draw_attribute_view_correlation_violin,
                                                                                            feat=feat,
                                                                                            # metric=self.individual_bias_metrics
                                                                                            ),
                                                                                    streams=[self.selected_nodes_stream,
                                                                                            self.individual_bias_metrics_stream]).opts(
                                                                                                                                height=int(self.diagnostic_panel_height*0.45),
                                                                                                                                width=int(self.diagnostic_panel_width*0.3),
                                                                                                                            )
                        else:
                            self.attribute_view_detail.object = hv.DynamicMap(pn.bind(draw.draw_attribute_view_violin,
                                                                                        variable_data=feat,
                                                                                        feat_name=feat_name),
                                                                                streams=[self.selected_nodes_stream, self.groups_stream]).opts(
                                                                                                                                                height=int(self.diagnostic_panel_height*0.45),
                                                                                                                                                width=int(self.diagnostic_panel_width*0.4),
                                                                                                                                            )
                            self.attribute_view_correlation.object = hv.DynamicMap(pn.bind(draw.draw_attribute_view_correlation_hex,
                                                                                            feat=feat,
                                                                                            # metric=self.individual_bias_metrics
                                                                                            ),
                                                                                    streams=[self.selected_nodes_stream,
                                                                                            self.individual_bias_metrics_stream]).opts(
                                                                                                                            height=int(self.diagnostic_panel_height*0.45),
                                                                                                                            width=int(self.diagnostic_panel_width*0.3),
                                                                                                                        )
                else:
                    self.attribute_view_detail.object = None
                    self.attribute_view_correlation.object = None
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
                            # tmp_ls = self.selected_attrs_ls_stream.selected_attrs_ls
                            # new_ls = tmp_ls[: -1] + [tmp_ls[-1] + [id]]
                            # self.selected_attrs_ls_stream.event(selected_attrs_ls=tmp_ls[:-1])
                
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
                            self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
                            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
                            tmp_contributions_selected_attrs = np.delete(contributions_selected_attrs, i)
                            self.contributions_stream.event(contributions_selected_attrs=tmp_contributions_selected_attrs)
                            break
        self.previous_attribute_view_overview_tap_x, self.previous_attribute_view_overview_tap_y = x, y

    def _update_selected_communities_dropdown(self, index): 
        # print('in _update_selected_communities_dropdown')  
        if index:
            self.selected_communities_dropdown.options = [str(list(self.node_communities[i]))[1:-1] for i in index]
        else:
            self.selected_communities_dropdown.options = [None] 
        self.selected_communities_dropdown.value = self.selected_communities_dropdown.options[0]

    def _update_subgraph_view(self, event):
        # print('in _update_subgraph_view')
        # if self.selected_communities_dropdown.value:
        #     # convert node_indices_selected to a list
        #     self.selected_nodes = np.array(list(map(int, self.selected_communities_dropdown.value.split(', '))))
        # else:
        #     # self.selected_nodes = self.node_indices
        #     # update self.selected_nodes
        #     mask = self._get_data_mask()
        #     self.selected_nodes = self.node_indices[mask]
        # if not np.array_equal(self.selected_nodes, self.previous_selected_nodes): # notice the order
        #     self.selected_nodes_stream.event(selected_nodes=self.selected_nodes)
        # self.previous_selected_nodes = self.selected_nodes
        if self.selected_communities_dropdown.value:
            # convert node_indices_selected to a list
            selected_nodes = np.array(list(map(int, self.selected_communities_dropdown.value.split(', '))))
        else:
            selected_nodes = self.node_indices
        self.selected_nodes_stream.event(selected_nodes=selected_nodes)

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
        # print('in _update_fairness_metric_detail')
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
                    chart.opts(width=int(self.fairness_metric_view_width*0.43), height=int(self.fairness_metric_view_height*0.77))
            else:
                if len(self.fairness_metric_view_chart_column) == 2:
                    self.fairness_metric_view_chart_column.pop(0)  
                chart.opts(width=int(self.fairness_metric_view_width*0.43), height=int(self.fairness_metric_view_height*0.95))

            self.fairness_metric_view_chart.object = chart


    def _min_threshold_slider_button_callback(self, event):
        thr = self.min_threshold_slider.value
        # Recalculate communities
        # self.node_communities = community.extract_communities(*copy.deepcopy(self.extract_communities_args), thr)
        # self.node_communities = community.extract_communities(community.Tree(), 
        #                                                       *(self.extract_communities_args[1: ]), thr)
        self.node_communities = community.extract_communities(community.Tree(), 
                                                              *copy.deepcopy(self.extract_communities_args[1: 3]), 
                                                              self.extract_communities_args[3], 
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

        # self.density_view_heatmap.object = draw.draw_density_view_heatmap(self.communities, self.min_threshold_slider.value)
        # self.data_subgraph_view.event(communities=self.communities)

    def _selected_nodes_stream_subscriber(self, selected_nodes):
        # create a np array containing 1 and 0.2 representing the alpha value of the nodes selected and not selected
        selected_alpha = np.zeros(self.n_nodes)
        selected_alpha[selected_nodes] = 1
        # not selected nodes are set to 0.2
        selected_alpha[selected_alpha == 0] = 0 
        # element-wise multiplication of the selected_alpha and the sampled_alpha
        # alpha = selected_alpha * self.data_graph_graph_view.sampled_alpha
        # alpha = selected_alpha * self.data_embedding_view.sampled_alpha
        alpha = selected_alpha * self.sampled_alpha
        # update the alpha of the nodes
        # self.graph_view_scatter.event(sampled_alpha=alpha)
        self.data_embedding_view_alpha.event(alpha=alpha)

        self._update_correlation_view()

        # update contributions
        contributions = util.calc_contributions(
            model=self.model,
            g=self.g,
            feat=self.feat,
            selected_nodes=self.selected_nodes_stream.selected_nodes,
            groups=self.groups_stream.groups)
        self.contributions_stream.event(contributions=contributions)

        # update self.contributions_selected_nodes_stream
        individual_bias_metrics = self.individual_bias_metrics_stream.individual_bias_metrics
        sic = individual_bias_metrics[self.selected_nodes_stream.selected_nodes].sum()
        gc = individual_bias.group_bias_contribution(
            self.adj,
            self.feat,
            self.model,
            self.groups_stream.groups,
            self.selected_nodes_stream.selected_nodes
        )
        self.contributions_selected_nodes_stream.event(contributions_selected_nodes=[sic, gc])

    def _contributions_selected_nodes_stream_subscriber(self, contributions_selected_nodes):
        # update correlation_view_md
        self.correlation_view_latex2.object = f'Separate: {(contributions_selected_nodes[0] * 100):.2f}%    Composite: {(contributions_selected_nodes[1] * 100):.2f}%'

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
        xy = self.data_embedding_view_xy.xy
        layer = self.layer_selection.value - 1
        groups = self.groups_stream.groups

        sens_selected = groups
        sens_selected_unique = np.unique(sens_selected)
        colormap = {sens_selected_unique[i]: self.colors[i] for i in range(len(sens_selected_unique))}

        min_thr, max_thr, polygons_lst, max_edges_lst = RangesetCategorical.pre_compute_contours(colormap, 
                                                                                                 xy, layer, groups)
        print('pre contours is fine')
        # update polygons_lst, max_edges_lst in the data_embedding_view
        # self.data_embedding_view.event(polygons_lst=polygons_lst, max_edges_lst=max_edges_lst)
        self.data_embedding_view_polys.event(polygons_lst=polygons_lst, max_edges_lst=max_edges_lst)
        self.data_embedding_view_thr_range.event(min_thr=min_thr, max_thr=max_thr)
        print('event done')

    def _update_thr_range(self, min_thr, max_thr):
        self.embedding_view_thr_slider.start = self.data_embedding_view_thr_range.min_thr
        self.embedding_view_thr_slider.end = self.data_embedding_view_thr_range.max_thr
        self.embedding_view_thr_slider.value = (self.data_embedding_view_thr_range.max_thr - self.data_embedding_view_thr_range.min_thr) / 10

    def _graph_view_scatter_selection1d_subscriber(self, index):
        # print('in _graph_view_scatter_selection1d_subscriber')
        # print(index)
        if index:
            if len(index) > 1:
                self.selected_nodes_stream.event(selected_nodes=np.array(index))
            else:
                idx = index[0]
                adj_mul_indices_current = self.adj_mul_indices[self.layer_selection.value - 1]
                selected_nodes = adj_mul_indices_current[1, adj_mul_indices_current[0] == idx].cpu().numpy()
                if idx not in selected_nodes:
                    selected_nodes = np.append(selected_nodes, idx)
                # print(selected_nodes) 
                self.selected_nodes_stream.event(selected_nodes=selected_nodes)
        else:
            self.selected_nodes_stream.event(selected_nodes=self.node_indices)

    def _selected_attrs_ls_stream_callback(self, selected_attrs_ls):
        # print('in _selected_attrs_ls_stream_callback')
        if selected_attrs_ls[-1]:
            # update contributions_stream.contributions_selected_attrs[-1] according to the selected_attrs_ls[-1]
            contribution_selected_attrs = util.calc_attr_contributions(
                model=self.model,
                g=self.g, 
                feat=self.feat,
                selected_nodes=self.selected_nodes_stream.selected_nodes,
                groups=self.groups_stream.groups,
                attr_indices=selected_attrs_ls[-1]
            )
            # print(contribution_selected_attrs) 
            # self.contributions_stream.contributions_selected_attrs[-1] = contribution_selected_attrs
            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
            contributions_selected_attrs[-1] = contribution_selected_attrs
        else:
            contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy() 
            contributions_selected_attrs[-1] = 0.
        # print(contributions_selected_attrs)
        self.contributions_stream.event(contributions_selected_attrs=contributions_selected_attrs) 
        # print(self.contributions_stream.contributions_selected_attrs)

    # def _correlation_view_scatter_selection1d_subscriber(self, index): 
    #     print('in _correlation_view_scatter_selection1d_subscriber')
    #     print(index)
    #     if index:
    #         self.selected_nodes_stream.event(selected_nodes=np.array(index))
    #     else:
    #         self.selected_nodes_stream.event(selected_nodes=self.node_indices)

    def _save_extract_communities_args(self):
        # Determine the directory of the current file
        current_dir = os.path.dirname(__file__)
        file_path = os.path.join(current_dir, 'extract_communities_args.pkl')

        # Use pickle to serialize self.extract_communities_args to the file
        with open(file_path, 'wb') as file:
            pickle.dump(self.extract_communities_args[1: 3], file) 

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
        # create a new list of selected attributes
        selected_attrs_ls = self.selected_attrs_ls_stream.selected_attrs_ls.copy()
        selected_attrs_ls.append([])
        self.selected_attrs_ls_stream.event(selected_attrs_ls=selected_attrs_ls)
        # append a 0. to self.contributions_stream.contributions_selected_attrs
        contributions_selected_attrs = self.contributions_stream.contributions_selected_attrs.copy()
        contributions_selected_attrs_tmp = np.append(contributions_selected_attrs, 0.)
        self.contributions_stream.event(contributions_selected_attrs=contributions_selected_attrs_tmp)

    def _record_button_callback(self, event):
        selected_nodes = self.selected_nodes_stream.selected_nodes
        selected_attrs = self.selected_attrs_ls_stream.selected_attrs_ls[-1] 
        self.records.append({'Nodes': selected_nodes, 'Attributes': selected_attrs})   

    def get_records(self):
        return self.records