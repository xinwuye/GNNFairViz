from . import util
from . import draw
import numpy as np
import holoviews as hv
from holoviews import opts
# import os
import torch
import numpy as np
from sklearn.manifold import TSNE
from holoviews.streams import Stream, param
from holoviews import streams
# import datashader as ds
# import datashader.transfer_functions as tf
# import pandas as pd
# from datashader.bundling import connect_edges, hammer_bundle
# from holoviews.operation.datashader import datashade, bundle_graph
import panel as pn
from .explainers import explain_unfairness_local
from .explainers import configs
# from . import RangesetCategorical
import copy
from bokeh.models import HelpButton, Tooltip

hv.extension('bokeh')

PERPLEXITY = 15 # german
SEED = 42

# set np seed
np.random.seed(SEED)


class DataGraphView(Stream):
    xy = param.Array(default=np.array([]), constant=False, doc='XY positions.')


class DataGraphGraphView(Stream):
    # numpy array
    source_id = param.Array(default=np.array([]), constant=True, doc='Source of edges.')
    target_id = param.Array(default=np.array([]), constant=True, doc='Target of edges.')
    # source_explained = param.Array(default=np.array([]), constant=True, doc='Source of explained edges.')
    # target_explained = param.Array(default=np.array([]), constant=True, doc='Target of explained edges.')
    sens = param.Array(default=np.array([]), constant=True, doc='Sensitive attributes.')
    sampled_alpha = param.Array(default=np.array([]), constant=True, doc='Sampled alpha.')
    # explain_induced_node_indices = param.Array(default=np.array([]), constant=True, doc='Induced node indices.')
    # explain_induced_node_hops = param.Array(default=np.array([]), constant=True, doc='Induced node hops.')
    sens_names = param.List(default=[], constant=True, doc='Sensitive attribute names.')
    bundle = param.Boolean(default=False, constant=True, doc='Whether to bundle edges.')


class DataLegend(Stream):
    sens = param.Array(default=np.array([]), constant=True, doc='Sensitive attributes.')
    sens_names = param.List(default=[], constant=True, doc='Sensitive attribute names.')


class DataExplanationView(Stream):
    xy = param.Array(default=np.array([]), constant=False, doc='XY positions.')
    source_explained = param.Array(default=np.array([]), constant=True, doc='Source of explained edges.')
    target_explained = param.Array(default=np.array([]), constant=True, doc='Target of explained edges.')
    explain_induced_node_indices = param.Array(default=np.array([]), constant=True, doc='Induced node indices.')
    explain_induced_node_indices_unfair_pure = param.Array(default=np.array([]), constant=True, doc='Induced unfair node indices.')
    explain_induced_node_indices_fair_pure = param.Array(default=np.array([]), constant=True, doc='Induced fair node indices.')
    explain_induced_node_indices_intersection = param.Array(default=np.array([]), constant=True, doc='Induced intersection node indices.')
    explain_induced_node_hops = param.Array(default=np.array([]), constant=True, doc='Induced node hops.')
    sens = param.Array(default=np.array([]), constant=True, doc='Sensitive attributes.')
    sens_names = param.List(default=[], constant=True, doc='Sensitive attribute names.')
    edge_type = param.List(default=[], constant=True, doc='Edge type.')


class DataMetricView(Stream):
    metrics = param.List(default=[], constant=True, doc='Metrics.')


# class DataAttributeView(Stream):
#     # hop_indicator_induced = param.List(default=[], constant=True, doc='Induced hop indicator.')
#     # hop_indicator_total = param.List(default=[], constant=True, doc='Total hop indicator.')
#     hop_indicator_induced_unfair = param.List(default=[], constant=True, doc='Unfair induced hop indicator.')
#     hop_indicator_induced_fair = param.List(default=[], constant=True, doc='Fair induced hop indicator.')
#     feat = param.Array(default=np.array([]), constant=True, doc='Features.')
#     max_hop = param.Integer(default=0, constant=True, doc='Max hop.')


# class DataAttrHist(Stream):
#     # hop_indicator_induced = param.List(default=[], constant=True, doc='Induced hop indicator.')
#     # hop_indicator_total = param.List(default=[], constant=True, doc='Total hop indicator.')
#     # feat = param.Array(default=np.array([]), constant=True, doc='Features.')
#     i_col = param.Integer(default=0, constant=True, doc='Column index.')


class DataFeatureView(Stream):
    feat = param.Array(default=np.array([]), constant=True, doc='Features.')
    hop_indicator_computation_graph = param.List(default=[], constant=True, doc='Computation graph.')
    hop_indicator_induced_unfair = param.List(default=[], constant=True, doc='Unfair induced hop indicator.')
    hop_indicator_induced_fair = param.List(default=[], constant=True, doc='Fair induced hop indicator.')
    # max_hop = param.Integer(default=0, constant=True, doc='Max hop.')


class EMUG:
    def __init__(self, model, adj, feat, sens, sens_names, max_hop, layers, perplexity=PERPLEXITY):
        self.model = copy.deepcopy(model)
        self.adj = adj
        self.feat = feat
        self.sens = sens
        self.sens_names = sens_names
        self.max_hop = max_hop

        # determine if each col is one-hot or not
        if len(feat.shape) > 2:
            feat_2d = feat.reshape(-1, feat.shape[-1])
        else:
            feat_2d = feat
        self.feat_types = np.all(np.logical_or(feat_2d == 0, feat_2d == 1), axis=0)

        # self.sub_adj = None
        self.neighbors = None

        preprocessed = util.init(model, adj, feat, sens, sens_names)
        self.embeddings = preprocessed['embeddings']
        self.degree_boxes = preprocessed['degree_boxes']
        self.metrics = util.calc_metrics(self.embeddings, sens_names, sens, self.degree_boxes)
        self.metric_types = list(self.metrics[0].columns)

        tsne = TSNE(perplexity=perplexity, 
            n_components=2, 
            init='pca', 
            n_iter=1250, 
            learning_rate='auto', random_state=SEED)
        self.embeddings_tsne = util.proj_emb(self.embeddings, tsne)

        # for et in self.embeddings_tsne:
        #     for sens_name in sens_names:
        #         sens_name_idx = sens_names.index(sens_name)
        #         sens_selected = sens[sens_name_idx]

        #         poly_df = RangesetCategorical.compute_contours(et, sens_selected, threshold=1)

        self.xy = np.array(self.embeddings_tsne)

        # self.masked_adj_unfair = all 0s with the same shape as self.adj
        self.masked_adj_unfair = np.zeros_like(adj)
        self.masked_adj_fair = np.zeros_like(adj)
        self.feat_mask_unfair = np.zeros(self.feat.shape[-1])
        self.feat_mask_fair = np.zeros(self.feat.shape[-1])
        self.layers = layers

        self.colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', 
            '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', 
            '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', 
            '#4e7670']
        
        self.selectable_metrics = []
        for layer in self.layers:
            for metric_type in self.metric_types:
                self.selectable_metrics.append(f'{layer}-{metric_type}')

        # sampled_alpha = 1000 1s
        self.n_nodes = adj.shape[0]
        sampled_alpha = np.zeros(self.n_nodes)
        self.init_sample_size = 200
        sample_idx = np.random.choice(np.arange(self.n_nodes), self.init_sample_size, replace=False)
        sampled_alpha[sample_idx] = 1
        # convert adj to pd.dataframe consisting of two cols, source and target representing the edges
        adj_triu = np.triu(adj, k=1)
        edges = adj_triu.nonzero()
        self.edges = np.array(edges).T
        sampled_edges = self.edges[np.isin(self.edges[:,0], sample_idx)]
        sampled_edges = sampled_edges[np.isin(sampled_edges[:,1], sample_idx)]
        np.random.shuffle(sampled_edges)
        self.init_edge_sample_size = 200
        if sampled_edges.shape[0] > self.init_edge_sample_size:
            sampled_edges = sampled_edges[: self.init_edge_sample_size]
        source = sampled_edges[:, 0]
        target = sampled_edges[:, 1]

        self.data_graph_view = DataGraphView(xy=self.xy)
        self.data_graph_graph_view = DataGraphGraphView(source_id=source, target_id=target, 
            sens=self.sens, sens_names=self.sens_names, sampled_alpha=sampled_alpha)
        self.data_legend = DataLegend(sens=self.sens, sens_names=self.sens_names)
        self.data_explanation_view = DataExplanationView(xy=self.xy, sens=self.sens, sens_names=self.sens_names)
        self.data_metric_view = DataMetricView(metrics=self.metrics)
        # self.data_attribute_view = DataAttributeView(feat=self.feat, max_hop=self.max_hop)
        # self.data_attr_hists = [DataAttrHist(i_col=i_col) for i_col in range(self.feat.shape[-1])]
        # self.data_feature_view = DataFeatureView(max_hop=self.max_hop, feat=self.feat)
        self.data_feature_view = DataFeatureView(feat=self.feat)
        # self.data_feature_view = DataFeatureViewTest(max_hop=self.max_hop, feat=self.feat)

        self.selection = streams.Selection1D()
        self.ids = self.selection.index
        self.sens_name = pn.widgets.Select(options=self.sens_names, name='Sensitive Attribute')
        self.threshold_slider = pn.widgets.FloatSlider(name='Threshold', start=0, end=1.0, step=0.001, value=0.999)

        self.graph_view = None
        self.explanation_view = None
        self.metric_view = None
        # self.attribute_view = None
        # self.attr_hists = []
        self.feature_view = None

    def show(self):
        self.explanation_hop_slider = pn.widgets.IntSlider(name='Explanation Hop', start=1, end=self.max_hop, step=1, value=self.max_hop)
        # widget for graph view
        self.layer = pn.widgets.Select(options=self.layers, name='Layer')
        bundle = pn.widgets.Checkbox(name='Bundle Edges', value=False)
        explain_button = pn.widgets.Button(name='Explain', button_type='default')    
        explain_button.on_click(self._explain_button_callback)
        self.node_sample_size_slider = pn.widgets.IntSlider(name='Node Sample Size', start=1, end=self.n_nodes, step=1, value=self.init_sample_size)
        self.edge_sample_size_slider = pn.widgets.IntSlider(name='Edge Sample Size', start=1, end=self.edges.shape[0], step=1, value=self.init_edge_sample_size)
        sample_button = pn.widgets.Button(name='Sample', button_type='default')
        sample_button.on_click(self._sample_button_callback)
        # widget for metric view
        selected_metrics = pn.widgets.MultiChoice(name='Metrics', value=[],
            options=self.selectable_metrics)
        # widget for explanation view
        layout = pn.widgets.Select(options=['tsne', 'hop-wise'], name='Layout', value='hop-wise')
        # widget for attribute view
        # n_bins = pn.widgets.IntSlider(name='# Bins', start=1, end=40, step=1, value=20)
        # max_hop = str(int(self.max_hop))
        # tooltip_attribute_view = Tooltip(content='upper: {}-hop explanation subgraph, lower: {}-hop subgraph)'.format(max_hop, max_hop), position="right")
        # help_button_attribute_view = HelpButton(tooltip=tooltip_attribute_view)

        self.graph_view = hv.DynamicMap(pn.bind(draw.draw_graph_view, layer=self.layer, sens_name=self.sens_name, bundle=bundle, colors=self.colors), 
            streams=[self.data_graph_graph_view, self.data_graph_view, self.selection]) \
            .opts(framewise=True, xaxis=None, yaxis=None)

        xdist = hv.DynamicMap(pn.bind(draw.draw_distribution, layer=self.layer, sens=self.sens, sens_names=self.sens_names, 
            sens_name=self.sens_name, colors=self.colors, x_or_y='x'),
            streams=[self.data_graph_view])

        ydist = hv.DynamicMap(pn.bind(draw.draw_distribution, layer=self.layer, sens=self.sens, sens_names=self.sens_names,
            sens_name=self.sens_name, colors=self.colors, x_or_y='y'),
            streams=[self.data_graph_view])

        legend = hv.DynamicMap(pn.bind(draw.draw_legend, sens_name=self.sens_name, colors=self.colors), streams=[self.data_legend])

        self.explanation_view = hv.DynamicMap(pn.bind(draw.draw_explanation_view, layer=self.layer, sens_name=self.sens_name, layout=layout, colors=self.colors),
            streams= [self.data_explanation_view])

        edge_legend = draw.draw_edge_legend()

        self.metric_view = hv.DynamicMap(pn.bind(draw.draw_metric_view, selected_metrics=selected_metrics), streams=[self.data_metric_view]) \
            .opts(framewise=True, legend_position='right', width=500, height=250)

        # self.attribute_view = hv.DynamicMap(pn.bind(draw.draw_attribute_view, n_bins=n_bins), streams=[self.data_attribute_view])
        # self.attr_hists = [hv.DynamicMap(pn.bind(draw.draw_attr_hist, n_bins=n_bins, colors=self.colors), streams=[data_attr_hist, self.data_attribute_view]) 
        #     for data_attr_hist in self.data_attr_hists]
        # self.attribute_view = pn.Row(scroll=True, width=600)
        # self.attribute_view.extend(self.attr_hists)

        # self.feature_view = hv.DynamicMap(pn.bind(draw.draw_feature_view, feat=self.feat), streams=[self.data_feature_view])
        self.feature_view = hv.DynamicMap(pn.bind(draw.draw_feature_view, max_hop=self.explanation_hop_slider), streams=[self.data_feature_view])
            
        app = pn.Column(
            pn.Row(
                pn.Column(
                    '### Control Panel', 
                    selected_metrics, 
                    self.layer, 
                    self.sens_name, 
                    self.threshold_slider, 
                    explain_button,
                    self.explanation_hop_slider,
                    ), 
                pn.Column(
                    '### Graph View', 
                    pn.Row(
                        bundle, 
                        self.node_sample_size_slider, 
                        self.edge_sample_size_slider,
                        sample_button, 
                        scroll=True,
                        width=400
                        ),
                    # (self.graph_view << ydist << xdist + legend).opts(shared_axes=False)
                    # (self.graph_view << ydist << xdist + legend).opts(shared_datasource=False)
                    # ((self.graph_view << ydist << xdist) + legend).opts(shared_axes=False)
                    self.graph_view << ydist << xdist
                    ), 
                pn.Column(
                    '### Explanation View', 
                    pn.Row(layout), 
                    (self.explanation_view + edge_legend).opts(shared_axes=False)
                    )
                ),
            pn.Row(
                pn.Column('### Metric View', self.metric_view), 
                pn.Column(
                    pn.Row(
                        # '### Attribute View',
                        '### Feature View',
                        # help_button_attribute_view
                        ), 
                    # self.attribute_view
                    self.feature_view
                    )
                )
            ).servable()
        return app

    # def _explain_button_callback_old(self, event):
    #     adj = self.adj
    #     max_hop = self.max_hop

    #     neighborhoods_max_hop = util.neighborhoods_each(adj, max_hop)
    #     if self.ids != self.selection.index:
    #         self._update_masked_adj_local(neighborhoods_max_hop)

    #     # create a masked_adj, in which the values other than adj[self.neighbors][: , self.neighbors] are set to 0
    #     # masked_adj = np.zeros_like(adj)
    #     # print('self.neighbors: ', self.neighbors)
    #     # masked_adj[self.neighbors][: , self.neighbors] = adj[self.neighbors][: , self.neighbors].copy()
    #     # masked_adj[np.ix_(self.neighbors, self.neighbors)] = adj[self.neighbors][: , self.neighbors].copy()
    #     # neighborhoods_max_hop_masked = util.neighborhoods_each(masked_adj, max_hop)
    #     # print('neighborhoods_max_hop_masked in _explain_button_callback: ', neighborhoods_max_hop_masked)

    #     threshold = self.threshold_slider.value
        
    #     indices = np.triu(self.masked_adj_unfair, k=1) > threshold
    #     sources, targets = np.where(indices)
    #     # print(indices)
    #     # get the unique values in the source and target
    #     explain_induced_node_indices = np.unique(np.concatenate([sources, targets]))
    #     sorter = np.argsort(explain_induced_node_indices)
    #     # create explain_induced_node_hops, an zero array with the same shape as explain_induced_node_indices
    #     explain_induced_node_hops = np.zeros_like(explain_induced_node_indices)

    #     # create a list of lists named hop_indicator in which each list contains the i-th hop neighbors of node whose index is id, the higher hop neighbors do not include the lower hop neighbors
    #     ids = self.ids
    #     hops = self.max_hop
    #     hop_indicator = []
    #     hop_indicator.append(ids)
    #     hop_indicator_masked = hop_indicator.copy()
    #     hop_indicator_total = hop_indicator.copy()
    #     hop_indicator_total_masked = hop_indicator.copy()
    #     for i in range(hops):
    #         current_neighborhoods = neighborhoods_max_hop[i]
    #         # current_neighborhoods_masked = neighborhoods_max_hop_masked[i]

    #         neighbors_adj_row = current_neighborhoods[ids, :].sum(dim=0)
    #         # neighbors_adj_row_masked = current_neighborhoods_masked[ids, :].sum(dim=0)

    #         tmp_indicater = neighbors_adj_row.nonzero().squeeze().tolist()
    #         hop_indicator_total.append(tmp_indicater)
            
    #         # tmp_indicater_masked = neighbors_adj_row_masked.nonzero().squeeze().tolist()
    #         # hop_indicator_total_masked.append(tmp_indicater_masked)

    #         # take the difference set of all the formers
    #         for j in range(i + 1):
    #             tmp_indicater = list(set(tmp_indicater) - set(hop_indicator[j]))
    #             # tmp_indicater_masked = list(set(tmp_indicater_masked) - set(hop_indicator_masked[j]))
    #         hop_indicator.append(tmp_indicater)
    #         # hop_indicator_masked.append(tmp_indicater_masked)
            
    #     hop_indicator_induced = hop_indicator_total.copy()
    #     # hop_indicator_induced_masked = hop_indicator_total_masked.copy()

    #     # hop_indicator_masked is a list of lists, find the slice of explain_induced_node_indices containing the values that are in one of the lists
    #     # explain_induced_node_indices_masked = np.unique(np.concatenate(hop_indicator_masked))
    #     # sorter_masked = np.argsort(explain_induced_node_indices_masked)
    #     # explain_induced_node_hops_masked = np.zeros_like(explain_induced_node_indices_masked)

    #     for i in range(hops + 1):
    #         # get the intersection of explain_induced_node_indices and hop_indicator[i]
    #         hop_indicator[i] = list(set(explain_induced_node_indices) & set(hop_indicator[i]))
    #         indices_hop = sorter[np.searchsorted(explain_induced_node_indices, hop_indicator[i], sorter=sorter)]
    #         explain_induced_node_hops[indices_hop] = i

    #         # hop_indicator_masked[i] = list(set(explain_induced_node_indices_masked) & set(hop_indicator_masked[i]))
    #         # indices_hop_masked = sorter_masked[np.searchsorted(explain_induced_node_indices_masked, hop_indicator_masked[i], sorter=sorter_masked)]
    #         # explain_induced_node_hops_masked[indices_hop_masked] = i

    #         hop_indicator_induced[i] = list(set(explain_induced_node_indices) & set(hop_indicator_induced[i]))
    #         # hop_indicator_induced_masked[i] = list(set(explain_induced_node_indices) & set(hop_indicator_induced_masked[i]))

    #     self.explanation_view.event(source_explained=sources, target_explained=targets,
    #         explain_induced_node_indices=explain_induced_node_indices,
    #         explain_induced_node_hops=explain_induced_node_hops)
    #         # explain_induced_node_indices=explain_induced_node_indices_masked,
    #         # explain_induced_node_hops=explain_induced_node_hops_masked)

    #     self.attr_hists[0].event(hop_indicator_total=hop_indicator_total, hop_indicator_induced=hop_indicator_induced)

    def _explain_button_callback(self, event):
        adj = self.adj
        # max_hop = self.max_hop
        max_hop = self.explanation_hop_slider.value

        neighborhoods_max_hop = util.neighborhoods_each(adj, max_hop)
        print('len of neighborhoods_max_hop: ', len(neighborhoods_max_hop))

        if self.ids != self.selection.index:
            self._update_masked_adj_local(neighborhoods_max_hop)
        
        print('feat_mask_unfair: ', self.feat_mask_unfair)
        print('feat_mask_fair: ', self.feat_mask_fair)

        threshold = self.threshold_slider.value
        
        indices_unfair = np.triu(self.masked_adj_unfair, k=1) > threshold
        indices_fair = np.triu(self.masked_adj_fair, k=1) > threshold
        indices = indices_unfair | indices_fair
        indices_unfair_subtracted = indices ^ indices_fair
        indices_fair_subtracted = indices ^ indices_unfair
        indices_intersection = indices_unfair & indices_fair
        edge_type_arr = np.zeros_like(indices, dtype=np.int8)
        # print('sum of indices_unfair_subtracted: ', indices_unfair_subtracted.sum())
        # print('sum of indices_fair_subtracted: ', indices_fair_subtracted.sum())
        # print('sum of indices_intersection: ', indices_intersection.sum())
        # print('sum of sum of 3: ', indices_unfair_subtracted.sum() + indices_fair_subtracted.sum() + indices_intersection.sum())
        # print('len of indices: ', indices.sum())
        # edge_type_arr[indices_fair_subtracted] = 2
        edge_type_arr[indices_unfair_subtracted] = 1
        edge_type_arr[indices_fair_subtracted] = 2
        edge_type_arr[indices_intersection] = 3
        edge_type_int = edge_type_arr[edge_type_arr > 0]
        # create a list with the correspondence to edge_type_int: {1: 'unfair', 2: 'fair', 3: 'unfair & fair'}
        # edge_type_str = ['unfair', 'fair', 'unfair & fair']
        # edge_type = [edge_type_str[i - 1] for i in edge_type_int] # edge color indicator

        sources, targets = np.where(indices)
        # get the unique values in the source and target
        explain_induced_node_indices = np.unique(np.concatenate([sources, targets]))
        sorter = np.argsort(explain_induced_node_indices)
        # create explain_induced_node_hops, an zero array with the same shape as explain_induced_node_indices
        explain_induced_node_hops = np.zeros_like(explain_induced_node_indices)

        # for the y-axis of the graph layout, separate the nodes into 3 groups: unfair, fair, and unfair & fair
        sources_unfair, targets_unfair = np.where(indices_unfair)
        sources_fair, targets_fair = np.where(indices_fair)
        explain_induced_node_indices_unfair = np.unique(np.concatenate([sources_unfair, targets_unfair]))
        explain_induced_node_indices_fair = np.unique(np.concatenate([sources_fair, targets_fair]))
        
        explain_induced_node_indices_unfair_pure = np.setdiff1d(explain_induced_node_indices_unfair, explain_induced_node_indices_fair)
        explain_induced_node_indices_fair_pure = np.setdiff1d(explain_induced_node_indices_fair, explain_induced_node_indices_unfair)
        explain_induced_node_indices_intersection = np.intersect1d(explain_induced_node_indices_unfair, explain_induced_node_indices_fair)

        sources_unfair, targets_unfair = np.where(indices_unfair)
        explain_induced_node_indices_unfair = np.unique(np.concatenate([sources_unfair, targets_unfair]))
        sources_fair, targets_fair = np.where(indices_fair)
        explain_induced_node_indices_fair = np.unique(np.concatenate([sources_fair, targets_fair]))

        # create a list of lists named hop_indicator in which each list contains the i-th hop neighbors of node whose index is id, the higher hop neighbors do not include the lower hop neighbors
        ids = self.ids
        # hops = self.max_hop
        hops = max_hop
        hop_indicator = []
        hop_indicator.append(ids)
        hop_indicator_total = hop_indicator.copy()
        for i in range(hops):
            current_neighborhoods = neighborhoods_max_hop[i]

            neighbors_adj_row = current_neighborhoods[ids, :].sum(dim=0)

            tmp_indicater = neighbors_adj_row.nonzero().squeeze().tolist()
            hop_indicator_total.append(tmp_indicater)

            # take the difference set of all the formers
            for j in range(i + 1):
                tmp_indicater = list(set(tmp_indicater) - set(hop_indicator[j]))
            hop_indicator.append(tmp_indicater)

        print('first hop_indicator[0]: ', hop_indicator[0])
            
        # hop_indicator_induced = hop_indicator_total.copy()
        hop_indicator_computation_graph = hop_indicator.copy()
        hop_indicator_induced_unfair = hop_indicator_total.copy()
        hop_indicator_induced_fair = hop_indicator_total.copy()

        print_counter = 0
        for i in range(hops + 1):
            # get the intersection of explain_induced_node_indices and hop_indicator[i]
            hop_indicator[i] = list(set(explain_induced_node_indices) & set(hop_indicator[i]))
            indices_hop = sorter[np.searchsorted(explain_induced_node_indices, hop_indicator[i], sorter=sorter)]
            explain_induced_node_hops[indices_hop] = i
            print_counter += len(indices_hop)

            # hop_indicator_induced[i] = list(set(explain_induced_node_indices) & set(hop_indicator_induced[i]))
            hop_indicator_induced_unfair[i] = list(set(explain_induced_node_indices_unfair) & set(hop_indicator_induced_unfair[i]))
            hop_indicator_induced_fair[i] = list(set(explain_induced_node_indices_fair) & set(hop_indicator_induced_fair[i]))
        print('print_counter: ', print_counter)
        print('explain_induced_node_indices: ', explain_induced_node_indices)
        print('hop_indicator[0]: ', hop_indicator[0])
        # print('len of explain_induced_node_indices_unfair: ', len(explain_induced_node_indices_unfair))
        # print('len of explain_induced_node_indices_fair: ', len(explain_induced_node_indices_fair))
        # print('type of self.explanation_view: ', type(self.explanation_view))
        # print('current key of self.explanation_view: ', self.explanation_view.current_key)
        # print('data of self.explanation_view: ', self.explanation_view.data)
        print('the node ids in 0 hop in emug: ', explain_induced_node_indices[explain_induced_node_hops == 0])
        self.explanation_view.event(source_explained=sources, target_explained=targets,
            explain_induced_node_indices=explain_induced_node_indices,
            explain_induced_node_indices_unfair_pure=explain_induced_node_indices_unfair_pure,
            explain_induced_node_indices_fair_pure=explain_induced_node_indices_fair_pure,
            explain_induced_node_indices_intersection=explain_induced_node_indices_intersection,
            explain_induced_node_hops=explain_induced_node_hops,
            edge_type=edge_type_int.tolist()
            # edge_type=edge_type
            )

        # self.attr_hists[0].event(hop_indicator_total=hop_indicator_total, hop_indicator_induced=hop_indicator_induced)
        # self.attr_hists[0].event(hop_indicator_induced_unfair=hop_indicator_induced_unfair, hop_indicator_induced_fair=hop_indicator_induced_fair)
        # print('type of self.feature_view: ', type(self.feature_view))
        # print('current key of self.feature_view: ', self.feature_view.current_key)
        # print('data of self.feature_view: ', self.feature_view.data)
        self.feature_view.event(
            hop_indicator_computation_graph=hop_indicator_computation_graph, 
            hop_indicator_induced_unfair=hop_indicator_induced_unfair, 
            hop_indicator_induced_fair=hop_indicator_induced_fair
            )

    def _sample_button_callback(self, event):
        node_sample_size = self.node_sample_size_slider.value
        edge_sample_size = self.edge_sample_size_slider.value

        sampled_alpha = np.zeros(self.n_nodes)
        sample_idx = np.random.choice(np.arange(self.n_nodes), node_sample_size, replace=False)
        sampled_alpha[sample_idx] = 1
        # convert adj to pd.dataframe consisting of two cols, source and target representing the edges
        sampled_edges = self.edges[np.isin(self.edges[:,0], sample_idx)]
        sampled_edges = sampled_edges[np.isin(sampled_edges[:,1], sample_idx)]
        np.random.shuffle(sampled_edges)
        if sampled_edges.shape[0] > edge_sample_size:
            sampled_edges = sampled_edges[: edge_sample_size]
        source = sampled_edges[:, 0]
        target = sampled_edges[:, 1]

        self.graph_view.event(sampled_alpha=sampled_alpha, source_id=source, target_id=target)


    def _update_masked_adj_local(self, neighborhoods_max_hop):
        self.ids = self.selection.index
        print('selected node ids: ', self.ids)
        
        if len(self.selection.index) > 0:
            # hops = self.max_hop
            hops = self.explanation_hop_slider.value
            sens_name_idx = self.sens_names.index(self.sens_name.value)

            # prog_args = configs.arg_parse_explain()
            prog_args = Args()
            prog_args.explainer_backbone = 'GNNExplainer'
            prog_args.num_gc_layers = hops

            sens = self.sens
            adj = torch.tensor(self.adj).float()
            # neighborhoods = neighborhoods_max_hop[hops - 1]
            neighborhoods = neighborhoods_max_hop[hops]
            neighborhoods = torch.where(neighborhoods != 0, torch.tensor(1), torch.tensor(0))

            feat = torch.tensor(self.feat).float()

            model = self.model
            # initialize explainer
            explainer = explain_unfairness_local.LocalExplainer(model=model, adj=adj, feat=feat, args=prog_args, neighborhoods=neighborhoods, layer=self.layer.value)

            self.masked_adj_unfair, self.masked_adj_fair, self.neighbors, self.feat_mask_unfair, self.feat_mask_fair = explainer.explain(sens[sens_name_idx], self.ids) # where the algorithm is implemented
        else:
            self.masked_adj_unfair = np.zeros_like(self.adj)
            self.masked_adj_fair = np.zeros_like(self.adj)
            self.feat_mask_unfair = np.zeros(self.feat.shape[-1])
            self.feat_mask_fair = np.zeros(self.feat.shape[-1])
            self.neighbors = None


class Args:
    explainer_backbone = 'GNNExplainer'
    num_gc_layers = None
    num_epochs = 1000
    gpu = False
    opt = 'adam'
    lr = 0.5
    opt_scheduler = 'none'
