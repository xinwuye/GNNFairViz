from . import util
from . import draw
from . import community
from .metrics import node_classification
from .metrics.pdd import pdd
import numpy as np
import holoviews as hv
from holoviews import opts
# import os
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

hv.extension('bokeh')
pn.extension()

PERPLEXITY = 15 # german
SEED = 42

# set np seed
np.random.seed(SEED)


class DataGraphView(Stream):
    xy = param.Array(default=np.array([]), constant=False, doc='XY positions.')
    sens_name = param.List(default=[], constant=True, doc='Sensitive attribute name.')


class DataGraphGraphView(Stream):
    # numpy array
    source_id = param.Array(default=np.array([]), constant=True, doc='Source of edges.')
    target_id = param.Array(default=np.array([]), constant=True, doc='Target of edges.')
    sens = param.Array(default=np.array([]), constant=True, doc='Sensitive attributes.')
    sampled_alpha = param.Array(default=np.array([]), constant=True, doc='Sampled alpha.')
    sens_names = param.List(default=[], constant=True, doc='Sensitive attribute names.')
    # bundle = param.Boolean(default=False, constant=True, doc='Whether to bundle edges.')


class DataLegend(Stream):
    sens = param.Array(default=np.array([]), constant=True, doc='Sensitive attributes.')
    sens_names = param.List(default=[], constant=True, doc='Sensitive attribute names.')


class DataMetricView(Stream):
    metrics = param.List(default=[], constant=True, doc='Metrics.')


class DataFeatureView(Stream):
    feat = param.Array(default=np.array([]), constant=True, doc='Features.')
    hop_indicator_computation_graph = param.List(default=[], constant=True, doc='Computation graph.')
    hop_indicator_induced_unfair = param.List(default=[], constant=True, doc='Unfair induced hop indicator.')
    hop_indicator_induced_fair = param.List(default=[], constant=True, doc='Fair induced hop indicator.')
    # max_hop = param.Integer(default=0, constant=True, doc='Max hop.')


class DataSubgraphView(Stream):
    communities = param.List(default=[], constant=True, doc='Communities.')


class EUG:
    def __init__(self, model, adj, feat, sens, sens_names, max_hop, masks, 
                 labels, emb_fnc, perplexity=PERPLEXITY, feat_names=None):
        # if feat_names is None
        if feat_names is None:
            self.feat_names = np.array([str(i) for i in range(feat.shape[-1])])
        else:
            self.feat_names = feat_names
        self.model = copy.deepcopy(model)        
        self.adj = adj
        self.adj0, self.adj1, self.adj0_scipy, self.adj1_scipy = util.modify_sparse_tensor_scipy(adj)
        self.feat = feat
        self.sens = sens
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
        model.eval()
        with torch.no_grad():
            logits = model(self.g, self.feat)
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
        self.embeddings_umap = []

        self.xy = np.array(self.embeddings_pca)
        # self.masked_adj_unfair = np.zeros_like(adj)
        # self.masked_adj_fair = np.zeros_like(adj)
        # self.feat_mask_unfair = np.zeros(self.feat.shape[-1])
        # self.feat_mask_fair = np.zeros(self.feat.shape[-1])
        self.layers = list(range(len(self.embeddings)))

        self.metric_name_dict = {'SD MF1': 'micro_f1_std_dev',
                        # 'DI': 'surrogate_di',
                        'EFPR': 'efpr',
                        'EFNR': 'efnr',
                        # 'ETPR': 'etpr',
                        'D SP': 'delta_dp_metric',
                        'Dm SP': 'max_diff_delta_dp',
                        'D EO': 'delta_eo_metric',
                        'D Acc': 'delta_accuracy_metric',
                        'SD Acc': 'sigma_accuracy_metric',
                        # 'D TED': 'delta_ted_metric',
                        }

        # self.colors = ['#d2bb4c', '#b054c1', '#82cd53', '#626ccc', 
        #     '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', 
        #     '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', 
        #     '#4e7670']
        self.colors = [
            '#d05d2e', '#6bd4a3', '#ce478c', '#5c7d39', '#bf4b56', 
            '#7ab7d5', '#a0714a', '#7b5d84', '#c2c6a4', '#d3a1c4', 
            '#4e7670']
        
        # self.selectable_metrics = []
        # for layer in self.layers:
        #     for metric_type in self.metric_types:
        #         self.selectable_metrics.append(f'{layer}-{metric_type}')

        # sampled_alpha = 1000 1s
        self.n_nodes = adj.shape[0]
        sampled_alpha = np.zeros(self.n_nodes)
        self.init_sample_size = 200
        sample_idx = np.random.choice(np.arange(self.n_nodes), self.init_sample_size, replace=False)
        sampled_alpha[sample_idx] = 1
        # convert adj to pd.dataframe consisting of two cols, source and target representing the edges
        # adj_triu = np.triu(adj, k=1)
        # edges = adj_triu.nonzero()
        # self.edges = np.array(edges).T
        indices_adj = adj.coalesce().indices()
        self.edges = indices_adj[: , indices_adj[0] > indices_adj[1]].cpu().numpy().T
        sampled_edges = self.edges[np.isin(self.edges[:,0], sample_idx)]
        sampled_edges = sampled_edges[np.isin(sampled_edges[:,1], sample_idx)]
        np.random.shuffle(sampled_edges)
        self.init_edge_sample_size = 200
        if sampled_edges.shape[0] > self.init_edge_sample_size:
            sampled_edges = sampled_edges[: self.init_edge_sample_size]
        source = sampled_edges[:, 0]
        target = sampled_edges[:, 1]

        # self.data_graph_view = DataGraphView(xy=self.xy)
        self.data_graph_graph_view = DataGraphGraphView(source_id=source, target_id=target, 
            sens=self.sens, sens_names=self.sens_names, sampled_alpha=sampled_alpha)
        self.data_legend = DataLegend(sens=self.sens, sens_names=self.sens_names)
        # self.data_metric_view = DataMetricView(metrics=self.metrics)
        # self.data_feature_view = DataFeatureView(feat=self.feat)

        # self.selection = streams.Selection1D()
        # self.ids = self.selection.index        
        # self.threshold_slider = pn.widgets.FloatSlider(name='Threshold', start=0, end=1.0, step=0.001, value=0.999)
        self.graph_view_scatter = None
        self.graph_view = None
        self.metric_view = None
        self.feature_view = None

    def show(self):
        height = 1900
        width = 900
        self.height = height
        self.width = width

        self.control_panel_height = int(height/2/3-35)
        self.control_panel_width = int(width*2/7)
        self.graph_view_height = int(height/2/3-35)
        self.graph_view_width = int(width*2/7)
        self.fairness_metric_view_height = int(height/2/3-35)
        self.fairness_metric_view_width = int(width*3/7)
        self.density_view_height = int(height/2/3-35)
        self.density_view_width = int(width*5/14)
        self.subgraph_view_height = int(height/2/3-35)
        self.subgraph_view_width = int(width*5/14)
        self.attribute_view_height = int(height/2/2-35)
        self.attribute_view_width = int(width*10/14)
        self.correlation_view_height = int(height/2*2/3-35)
        self.correlation_view_width = int(width*10/14)
        self.neighbor_view_height = int(height/2/3-35)
        self.neighbor_view_width = int(width/2)

        # widget for all
        # self.sens_name = pn.widgets.Select(options=self.sens_names, value = self.sens_names[0], name='Sensitive Attribute')
        self.sens_name = pn.widgets.MultiChoice(name='Sensitive Attribute', options=self.sens_names, value = [self.sens_names[0], ], width=200)
        self.sens_name_confirm_control_panel_button = pn.widgets.Button(name='Confirm', button_type='default')
        self.sens_name_confirm_control_panel_button.on_click(self._sens_name_confirm_control_panel_button_callback)

        # widget for graph view
        self.layer = pn.widgets.Select(options=self.layers, name='Layer', width=200)
        bundle = pn.widgets.Checkbox(name='Bundle Edges', value=False)
        self.node_sample_size_slider = pn.widgets.IntSlider(name='Node Sample Size', start=1, end=self.n_nodes, step=1, value=self.init_sample_size, width=200)
        self.edge_sample_size_slider = pn.widgets.IntSlider(name='Edge Sample Size', start=1, end=self.edges.shape[0], step=1, value=self.init_edge_sample_size, width=200)
        sample_button = pn.widgets.Button(name='Sample', button_type='default')
        sample_button.on_click(self._sample_button_callback)

        # widget for metric view
        # selected_metrics = pn.widgets.MultiChoice(name='Metrics', value=[],
        #     options=self.selectable_metrics)

        # widget for fairness metric view
        # a multi-select widget for selecting "train", "val", "test", "other"
        self.data_selector = pn.widgets.MultiChoice(name='Data', value=['train', 'val', 'test', 'other'],
            options=['train', 'val', 'test', 'other'],
            width=200,
            )
        data_confirm_fairness_metric_view_button = pn.widgets.Button(name='Confirm', button_type='default')
        data_confirm_fairness_metric_view_button.on_click(self._data_confirm_fairness_metric_view_button_callback)

### graph view
        # wh_graph_view = min(width/3, height/2-115)
        wh_graph_view = min(self.graph_view_width, self.graph_view_height-35)
        self.data_graph_view = DataGraphView(xy=self.xy, sens_name=self.sens_name.value)

        self.graph_view_scatter = hv.DynamicMap(pn.bind(draw.draw_graph_view, layer=self.layer, bundle=bundle, colors=self.colors), 
            # streams=[self.data_graph_graph_view, self.data_graph_view, self.selection]) \
            streams=[self.data_graph_graph_view, self.data_graph_view]) \
            .opts(framewise=True, 
                  xaxis=None, yaxis=None, 
                  width=int(wh_graph_view*0.75), 
                  height=int(wh_graph_view*0.75),
                  shared_axes=False,
                  )

        self.graph_view_xdist = hv.DynamicMap(pn.bind(draw.draw_distribution, layer=self.layer, sens=self.sens, sens_names=self.sens_names, 
            colors=self.colors, x_or_y='x'),
            streams=[self.data_graph_view]) \
            .opts(height=int(wh_graph_view*0.25))

        self.graph_view_ydist = hv.DynamicMap(pn.bind(draw.draw_distribution, layer=self.layer, sens=self.sens, sens_names=self.sens_names,
            colors=self.colors, x_or_y='y'),
            streams=[self.data_graph_view]) \
            .opts(width=int(wh_graph_view*0.25))
        
        self.graph_view = pn.Card(
            self.graph_view_scatter << self.graph_view_ydist << self.graph_view_xdist,
            hide_header=True,
            name='Graph View',
            # height=int(height/2-80), 
            # width=int(width/3),
            height=self.graph_view_height,
            width=self.graph_view_width,
            )
        
        self.graph_view_widgets = pn.WidgetBox(
            self.layer, 
            bundle,
            self.node_sample_size_slider,
            self.edge_sample_size_slider,
            sample_button,
            # width=int(width/3), 
            # height=int(height/2-80),
            height=self.graph_view_height,
            width=self.graph_view_width,
            name='Settings',
            )

        # legend = hv.DynamicMap(pn.bind(draw.draw_legend, sens_name=self.sens_name, colors=self.colors), streams=[self.data_legend])

        # edge_legend = draw.draw_edge_legend()

### fairness metric view
        self.fairness_metric_view_chart = pn.pane.HoloViews()
        # Create buttons and labels
        labeled_mask = self.labels != -1
        labeled_predictions = self.predictions[labeled_mask]
        labeled_labels = self.labels[labeled_mask]
        labeled_sens = self.sens[0][labeled_mask]
        letter_values = {'SD MF1': [node_classification.micro_f1_std_dev(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                        #  'DI': node_classification.surrogate_di(labeled_predictions, labeled_labels, labeled_sens),
                         'EFPR': [node_classification.efpr(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                         'EFNR': [node_classification.efnr(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                        #  'ETPR': node_classification.etpr(labeled_predictions, labeled_labels, labeled_sens),
                         'D SP': [node_classification.delta_dp_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                         'Dm SP': [node_classification.max_diff_delta_dp(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
                         'D EO': [node_classification.delta_eo_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                         'D Acc': [node_classification.delta_accuracy_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 1.0)],
                         'SD Acc': [node_classification.sigma_accuracy_metric(labeled_predictions, labeled_labels, labeled_sens), (0.0, 0.5)],
                        #  'D TED': node_classification.delta_ted_metric(labeled_predictions, labeled_labels, labeled_sens),
                         }
        # self.fairness_metric_view_labels = []
        self.items = []
        self.metric_widgets = []
        for letter, [value, bounds] in letter_values.items():
            # button = pn.widgets.Button(name=letter, width=100)
            # button = pn.widgets.Button(name='🔍', height=20)
            button = pn.widgets.Button(name=letter, height=20, width=60)
            # button.on_click(self._draw_fairness_metric_detail)
            pn.bind(self._draw_fairness_metric_detail, button, metric_name=letter, watch=True)
            # label = pn.widgets.StaticText(value='{:.2f}'.format(value))
            label = pn.indicators.LinearGauge(
                name=letter, value=value, bounds=bounds, format='{value:.2f}',
                horizontal=True, width=20, height=40, title_size='3px', 
                )
            # self.fairness_metric_view_labels.append(label)
            self.items.append(pn.Row(pn.Column(button, width=60), pn.Column(label), width=105))
            self.metric_widgets.append(label)

        self.fairness_metric_view_widgets = pn.WidgetBox(
            pn.Row(
                self.data_selector,
                data_confirm_fairness_metric_view_button
                ),
            # width=int(width/3), 
            # height=int(height/2-80),
            height=self.fairness_metric_view_height,
            width=self.fairness_metric_view_width,
            name='Settings',
            )
        
        self.fairness_metric_view = pn.Card(
            pn.Row(
                pn.Column(*self.items), 
                self.fairness_metric_view_chart
                ),
            hide_header=True,
            name='Fairness Metric View',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.fairness_metric_view_height,
            width=self.fairness_metric_view_width,
        )

### correlation view
        idx_train = self.train_mask.nonzero().flatten()
        idx_test = self.test_mask.nonzero().flatten()
        idx_val = self.val_mask.nonzero().flatten()
        estimated_pdd = pdd(self.adj, 
                            idx_train, 
                            self.feat, 
                            idx_test, 
                            torch.from_numpy(self.labels), 
                            self.sens[0], 
                            idx_val, 
                            self.model, 
                            self.g)
        feat_train = self.feat[self.train_mask].cpu().numpy()
        # print(feat_train.shape)
        # Check if each column consists of only 0s and 1s
        # columns_only_0_and_1 = np.all(np.logical_or(feat_train == 0, feat_train == 1), axis=0)
        self.columns_categorical = util.check_unique_values(feat_train)
        # correlation_matrix = np.corrcoef(feat_train.T, estimated_pdd)
        pearson_r_results = [stats.pearsonr(row, estimated_pdd) for row in feat_train.T[~self.columns_categorical]]
        pearson_correlations = [r[0] for r in pearson_r_results]
        pearson_p_vals = [r[1] for r in pearson_r_results]

        pointbiserialr_results = [stats.pointbiserialr(row, estimated_pdd) for row in feat_train.T[self.columns_categorical]]
        pointbiserial_correlations = [r[0] for r in pointbiserialr_results]
        pointbiserial_p_vals = [r[1] for r in pointbiserialr_results]
        
        height_correlation_view_selection_continuous = 40
        height_correlation_view_selection_categorical = height_correlation_view_selection_continuous
        width_correlation_view_selection_continuous = height_correlation_view_selection_continuous * len(pearson_correlations)
        width_correlation_view_selection_categorical = height_correlation_view_selection_continuous * len(pointbiserial_correlations)

        self.correlation_view_selection_continuous = draw.draw_correlation_view_selection(pearson_correlations, pearson_p_vals, self.feat_names[~self.columns_categorical]) \
            .opts(opts.Rectangles(height=height_correlation_view_selection_continuous, width=width_correlation_view_selection_continuous))
        self.correlation_view_hex = hv.DynamicMap(pn.bind(draw.draw_correlation_view_hex, feat=feat_train.T[~self.columns_categorical], pdd=estimated_pdd),
                                                  streams=[streams.SingleTap(source=self.correlation_view_selection_continuous)]) \
            .opts(width=int(self.correlation_view_width*0.4), 
                  height=int(self.correlation_view_height*0.6),
                  shared_axes=False,
                  )
        self.correlation_view_selection_categorical = draw.draw_correlation_view_selection(pointbiserial_correlations, pointbiserial_p_vals, self.feat_names[self.columns_categorical]) \
            .opts(opts.Rectangles(height=height_correlation_view_selection_categorical, width=width_correlation_view_selection_categorical))
        self.correlation_view_violin = hv.DynamicMap(pn.bind(draw.draw_correlation_view_violin, feat=feat_train.T[self.columns_categorical], pdd=estimated_pdd),
                                                    streams=[streams.SingleTap(source=self.correlation_view_selection_categorical)]) \
            .opts(width=int(self.correlation_view_width*0.4),
                    height=int(self.correlation_view_height*0.6),
                    shared_axes=False,
                    )
        correlation_view_legend = draw.draw_correlation_view_legend() \
            .opts(width=int(self.correlation_view_width*0.8),
                    height=40,
                    shared_axes=False,
                    )

        self.correlation_view = pn.Card(
            pn.Row(
                # pn.Column(
                #     self.correlation_view_selection_continuous, 
                #     self.correlation_view_selection_categorical,
                #     sizing_mode='stretch_height',
                # ),
                self.correlation_view_selection_continuous, 
                # sizing_mode='stretch_height', 
                scroll=True, 
                width=self.correlation_view_width
            ),
            pn.Row(
                self.correlation_view_selection_categorical, 
                sizing_mode='stretch_height', 
                scroll=True, 
                width=self.correlation_view_width
            ),
            correlation_view_legend,
            pn.Row(
                self.correlation_view_hex,
                self.correlation_view_violin,
            ),
            hide_header=True,
            name='Correlation View',
            height=self.correlation_view_height,
            width=self.correlation_view_width,
            )

### attribute view
        # continuous attributes
        feat_continuous = self.feat[:, ~self.columns_categorical].cpu().numpy()
        feat_names_continuous = self.feat_names[~self.columns_categorical]
        feat_continuous_df = pd.DataFrame(feat_continuous, columns=feat_names_continuous)
        # Initialize an empty list to hold the plots
        self.attribute_view_violins = []
        # Iterate over each feature name in the DataFrame
        for feat_name in feat_names_continuous:
            # Extract the column (variable) data
            variable_data = feat_continuous_df[feat_name]          
            # Generate the violin plot for this variable
            plot = draw.draw_attribute_view_violin(variable_data, feat_name, self.sens[0])   
            # Append the generated plot to the list of plots
            self.attribute_view_violins.append(plot)
        # Combine all plots vertically into a single layout
        self.attribute_view_violin = hv.Layout(self.attribute_view_violins).cols(1)

        # categorical attributes
        feat_categorical = self.feat[:, self.columns_categorical].cpu().numpy()
        feat_names_categorical = self.feat_names[self.columns_categorical]
        feat_categorical_df = pd.DataFrame(feat_categorical, columns=feat_names_categorical)
        # Initialize an empty list to hold the bar charts
        self.attribute_view_bars = []
        # Iterate over each feature name in the DataFrame
        for feat_name in feat_names_categorical:
            # Extract the column (variable) data
            variable_data = feat_categorical_df[feat_name]            
            # Generate the bar chart for this variable
            bar_chart = draw.draw_attribute_view_bar(variable_data, feat_name, self.sens[0])            
            # Append the generated chart to the list
            self.attribute_view_bars.append(bar_chart)
        # Combine all bar charts vertically into a single layout
        self.attribute_view_bar = hv.Layout(self.attribute_view_bars).cols(1)

        # sensitive attribute
        self.attribute_view_sens = draw.draw_attribute_view_sens(self.sens[0])
        bar_chart = draw.draw_attribute_view_bar(variable_data, feat_name, self.sens[0]) 

        self.attribute_view = pn.Card(
            hv.Layout([self.attribute_view_sens, ]).cols(1),
            pn.Row(
                pn.Column(
                    self.attribute_view_violin, 
                    scroll=True, 
                    height=int(self.attribute_view_height*0.6)
                ),
                pn.Column(
                    self.attribute_view_bar, 
                    scroll=True, 
                    height=int(self.attribute_view_height*0.6)
                ),  
            ),
            hide_header=True,
            name='Attribute View',
            height=self.attribute_view_height,
            width=self.attribute_view_width,
            )

### density view
        # Create the scatter plot
        self.extract_communities_args = community.process_graph(self.adj0)
        node_communities = community.extract_communities(*self.extract_communities_args, 0.5)
        self.communities = []
        for indices in node_communities:
            # Convert set of indices to sorted list for slicing
            indices_sorted = sorted(list(indices))
            # Slice rows: Efficient in CSR format
            row_sliced = self.adj0_scipy[indices_sorted, :]
            # Slice columns: Convert to CSC format for efficient column slicing if necessary
            # For simplicity, here we use the CSR format (less efficient for cols)
            final_slice = row_sliced[:, indices_sorted]
            self.communities.append(final_slice)
        self.data_subgraph_view = DataSubgraphView(communities=self.communities)
        graph_metrics = util.calculate_graph_metrics(self.communities) # Adjust this line as per your actual data structure
        self.density_view_scatter = draw.draw_density_view_scatter(graph_metrics) \
            .opts(width=int(self.density_view_width), 
                  height=int(self.density_view_height),
                  shared_axes=False,
                  )

        # Slider for min_threshold
        self.min_threshold_slider = pn.widgets.FloatSlider(name='Minimum Density Threshold', start=0.0, end=1.0, step=0.01, value=0.5)

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
            self.density_view_scatter,
            hv.streams.Selection1D(source=self.density_view_scatter),
            hide_header=True,
            name='Density View',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.density_view_height,
            width=self.density_view_width,
            )
        
### subgraph view
        # Create the heatmap (initially with index 0)
        self.subgraph_view_heatmap = hv.DynamicMap(draw.draw_subgraph_view_heatmap, 
                                                  streams=[hv.streams.Selection1D(source=self.density_view_scatter), 
                                                           self.data_subgraph_view]) \
            .opts(width=int(self.subgraph_view_width), 
                  height=int(self.subgraph_view_height),
                  shared_axes=False,
                  )
        
        self.subgraph_view = pn.Card(
            self.subgraph_view_heatmap,
            hide_header=True,
            name='Subgraph View',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.subgraph_view_height,
            width=self.subgraph_view_width,
            )
        
        # neighbor view
        self.neighbor_view = pn.Card(
            pn.pane.HoloViews(),
            hide_header=True,
            name='Neighbor View',
            # height=int(height/2-80),
            # width=int(width/3),
            height=self.neighbor_view_height,
            width=self.neighbor_view_width,
            )
        
        app = pn.GridSpec(width=self.width, height=self.height)
        app[0: 2, 0: 4] = pn.Tabs(
            pn.WidgetBox(
                self.sens_name, 
                self.sens_name_confirm_control_panel_button,
                name='Control Panel',
                # height=int(height/2-80), width=int(width/3),
                height=self.control_panel_height, 
                width=self.control_panel_width,
            ),
            # height=int(height/2), 
            # width=int(width/3),
            # height=self.control_panel_height+80,
            # width=self.control_panel_width,
            )
        app[0: 2, 4: 8] = pn.Tabs(
            self.graph_view,
            self.graph_view_widgets, 
            # height=int(height/2), 
            # width=int(width/3),
            # height=self.graph_view_height+80,
            # width=self.graph_view_width,
            )
        app[0: 2, 8: 14] = pn.Tabs(
            self.fairness_metric_view,
            self.fairness_metric_view_widgets,
            # height=int(height/2), 
            # width=int(width/3),
            # height=self.fairness_metric_view_height+80,
            # width=self.fairness_metric_view_width,
            )
        app[2: 5, 10: 14] = pn.Tabs(
            self.density_view,
            self.density_view_widgets,
            )
        app[6: 9, 10: 14] = pn.Tabs(
            self.subgraph_view,
            )
        app[6: 11, 0: 10] = pn.Tabs(
            self.attribute_view,
            )
        app[2: 6, 0: 10] = pn.Tabs(
            self.correlation_view,
            # height=self.density_view_height+80,
            # width=self.density_view_width,
            )
        # app[2, 7: 14] = pn.Tabs(
        #     self.neighbor_view,
        #     # height=self.neighbor_view_height+80,
        #     # width=self.neighbor_view_width,
        #     )

        return app

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

        self.graph_view_scatter.event(sampled_alpha=sampled_alpha, source_id=source, target_id=target)

    def _data_confirm_fairness_metric_view_button_callback(self, event):
        # Find the indices of the selected sensitive attributes
        sens_indices = [self.sens_names.index(name) for name in self.sens_name.value]
        sens_selected = self.sens[sens_indices]
        vectorized_concat = np.vectorize(util.concatenate_elements)
        # Apply the vectorized function across columns
        sens = vectorized_concat(*sens_selected)

        # Update the metrics
        updated_metrics = self._update_fairness_metrics(sens)

        # Update the labels with new metric values
        for metric_widget, metric_name in zip(self.metric_widgets, updated_metrics.keys()):
            metric_widget.value = updated_metrics[metric_name]

    def _update_fairness_metrics(self, sens):
        """
        Update the fairness metrics based on the given sensitive attribute.
        """
        mask = self._get_data_mask()
        predictions = self.predictions[mask]
        labels = self.labels[mask]
        labeled_mask = labels != -1
        predictions = predictions[labeled_mask]
        labels = labels[labeled_mask]
        sens = sens[labeled_mask]

        # Recalculate metrics
        # You can add more metrics as per your requirement
        metrics = {
            'SD MF1': node_classification.micro_f1_std_dev(predictions, labels, sens),
            'EFPR': node_classification.efpr(predictions, labels, sens),
            'EFNR': node_classification.efnr(predictions, labels, sens),
            'D SP': node_classification.delta_dp_metric(predictions, labels, sens),
            'Dm SP': node_classification.max_diff_delta_dp(predictions, labels, sens),
            'D EO': node_classification.delta_eo_metric(predictions, labels, sens),
            'D Acc': node_classification.delta_accuracy_metric(predictions, labels, sens),
            'SD Acc': node_classification.sigma_accuracy_metric(predictions, labels, sens),
        }

        return metrics

    def _get_data_mask(self):
        """
        Create a mask for the selected data categories (train, val, test, other).
        """
        mask = torch.zeros_like(self.train_mask).bool()
        if 'train' in self.data_selector.value:
            mask = mask | self.train_mask
        if 'val' in self.data_selector.value:
            mask = mask | self.val_mask
        if 'test' in self.data_selector.value:
            mask = mask | self.test_mask
        if 'other' in self.data_selector.value:
            mask = mask | self.other_mask
        return mask

    def _sens_name_confirm_control_panel_button_callback(self, event):
        sens_name = self.sens_name.value
        self.graph_view_scatter.event(sens_name=sens_name)
        self.graph_view_xdist.event(sens_name=sens_name)
        self.graph_view_ydist.event(sens_name=sens_name)
        self._data_confirm_fairness_metric_view_button_callback(event)

    def _draw_fairness_metric_detail(self, event, metric_name):
        mask = torch.zeros_like(self.train_mask).bool()
        if 'train' in self.data_selector.value:
            mask = mask | self.train_mask
        if 'val' in self.data_selector.value:
            mask = mask | self.val_mask
        if 'test' in self.data_selector.value:
            mask = mask | self.test_mask
        if 'other' in self.data_selector.value:
            mask = mask | self.other_mask

        predictions = self.predictions[mask]
        labels = self.labels[mask]
        tmp_sens = self.sens[:, mask]
        # sens = self.sens[0][mask]
        # sens = self.sens[self.sens_names.index(self.sens_name.value)][mask]

        # find the index of sens_name in sens_names
        sens_name_idx = []
        for sn in self.sens_name.value:
            sn_idx = self.sens_names.index(sn)
            sens_name_idx.append(sn_idx)

        tmp_sens_selected = tmp_sens[sens_name_idx]
        # Vectorize the function
        vectorized_concat = np.vectorize(util.concatenate_elements)
        # Apply the vectorized function across columns
        sens = vectorized_concat(*tmp_sens_selected)

        labeled_mask = labels != -1
        predictions = predictions[labeled_mask]
        labels = labels[labeled_mask]
        groups = sens[labeled_mask]
        unique_labels = np.unique(labels)
        unique_groups = np.unique(groups)
        # get the name of the clicked button
        # metric_name = event.obj.name
        if self.metric_name_dict[metric_name] == 'micro_f1_std_dev':
            micro_f1_scores = []
            for group in unique_groups:
                group_indices = np.where(groups == group)
                group_pred = predictions[group_indices]
                group_labels = labels[group_indices]

                micro_f1 = f1_score(group_labels, group_pred, average='micro')
                micro_f1_scores.append(micro_f1)
            data = {'Sensitive Subgroup': unique_groups, 'Micro-F1': micro_f1_scores}
            chart = draw.draw_bar_metric_view(data, 'Micro-F1')
        # elif self.metric_name_dict[metric_name] == 'surrogate_di':
        #     heatmap_data = []

        #     for i, label in enumerate(unique_labels):
        #         for j, group in enumerate(unique_groups):
        #             group_indices = np.where(groups == group)
        #             group_predictions = predictions[group_indices]

        #             # Proportion of positive predictions for the current group and label
        #             positive_proportion = np.mean(group_predictions == label)
        #             heatmap_data.append((str(int(label)), group, positive_proportion))

        #     # Generate the heatmap using the draw module
        #     chart = draw.draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'Pos. Pred. Rate')
        elif self.metric_name_dict[metric_name] == 'efpr':
            heatmap_data = []
            
            for label in unique_labels:
                for group in unique_groups:
                    group_indices = np.where(groups == group)
                    group_pred = predictions[group_indices]
                    group_labels = labels[group_indices]

                    tn = np.sum((group_pred != label) & (group_labels != label))
                    fp = np.sum((group_pred == label) & (group_labels != label))
                    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                    heatmap_data.append((str(int(label)), group, fpr))
            
            chart = draw.draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'FPR')
        elif self.metric_name_dict[metric_name] == 'efnr':
            heatmap_data = []
            
            for label in unique_labels:
                for group in unique_groups:
                    group_indices = np.where(groups == group)
                    group_pred = predictions[group_indices]
                    group_labels = labels[group_indices]

                    fn = np.sum((group_pred != label) & (group_labels == label))
                    tp = np.sum((group_pred == label) & (group_labels == label))
                    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

                    heatmap_data.append((str(int(label)), group, fnr))
            
            chart = draw.draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'FNR')
        elif self.metric_name_dict[metric_name] == 'delta_dp_metric' or self.metric_name_dict[metric_name] == 'max_diff_delta_dp':
            heatmap_data = []

            for label in unique_labels:
                for group in unique_groups:
                    group_indices = np.where(groups == group)
                    group_predictions = predictions[group_indices]

                    prob = np.mean(group_predictions == label)
                    heatmap_data.append((str(int(label)), group, prob))

            chart = draw.draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'P(Pred.=Label)')
        elif self.metric_name_dict[metric_name] == 'delta_eo_metric':
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
            
            chart = draw.draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'ETPR')
        elif self.metric_name_dict[metric_name] == 'delta_accuracy_metric' or self.metric_name_dict[metric_name] == 'sigma_accuracy_metric':
            group_accs = []
            for group in unique_groups:
                group_indices = np.where(groups == group)
                group_accuracy = accuracy_score(labels[group_indices], predictions[group_indices])

                group_accs.append(group_accuracy)

            data = {'Sensitive Subgroup': unique_groups, 'Accuracy': group_accs}
            chart = draw.draw_bar_metric_view(data, 'Accuracy')
        elif self.metric_name_dict[metric_name] == 'delta_ted_metric':
            heatmap_data = []

            for label in unique_labels:
                for group in unique_groups:
                    group_indices = np.where(groups == group)
                    fp = np.sum((predictions[group_indices] == label) & (labels[group_indices] != label))
                    fn = np.sum((predictions[group_indices] != label) & (labels[group_indices] == label))

                    ratio = fp / fn if fn > 0 else float('inf')
                    heatmap_data.append((str(int(label)), group, ratio))

            chart = draw.draw_heatmap_metric_view(heatmap_data, 'Label', 'Sensitive Subgroup', 'FP/FN Ratio')

        # chart.opts(width=int(self.width/3-120), height=int(self.height/2-115))
        chart.opts(width=int(self.fairness_metric_view_width-120), height=int(self.fairness_metric_view_height-35))

        self.fairness_metric_view_chart.object = chart

    def _min_threshold_slider_button_callback(self, event):
        # Recalculate communities
        self.communities = community.extract_communities(*self.extract_communities_args, 0.5)
        graph_metrics = util.calculate_graph_metrics(self.communities)
        self.density_view_scatter.object = draw.draw_density_view_scatter(graph_metrics)

        # self.density_view_heatmap.object = draw.draw_density_view_heatmap(self.communities, self.min_threshold_slider.value)
        self.data_subgraph_view.event(communities=self.communities)