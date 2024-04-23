import numpy as np
import holoviews as hv
# from holoviews import opts
import os
import torch
import numpy as np
# from sklearn.manifold import TSNE
# from holoviews.streams import Stream, param
# from holoviews import streams
# import datashader as ds
# import datashader.transfer_functions as tf
# import pandas as pd
# from datashader.bundling import connect_edges, hammer_bundle
# from holoviews.operation.datashader import datashade, bundle_graph
# import panel as pn
import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from dgl import AddSelfLoop
# from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

# for metrics
# from sklearn.metrics import f1_score
# from sklearn.metrics import accuracy_score
import numpy as np
import sys
sys.path.append('../../..')
from pyeug import eug
sys.path.remove('../../..')
import panel as pn
pn.extension()

hv.extension('bokeh')
class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        assert num_layers >= 2, "Number of layers should be at least 2."
        # first layer
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        # hidden layers
        for _ in range(1, num_layers-1):
            self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu))
        # output layer
        self.layers.append(dglnn.GraphConv(hid_size, out_size))
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:  # apply dropout after the first layer
                h = self.dropout(h)
            h = layer(g, h)
        return h
def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1
sys.path.append('../..')
from datasets import Bail
bail = Bail()
adj, features, idx_train, idx_val, idx_test, labels, sens, feat_names, sens_names \
    = bail.adj(), bail.features(), bail.idx_train(), bail.idx_val(), \
      bail.idx_test(), bail.labels(), bail.sens(), bail.feat_names(), bail.sens_names()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# masks = g.ndata["train_mask"], g.ndata["val_mask"], g.ndata["test_mask"]
# Get the source and destination node IDs
# adj = adj.int().to(device)
src, dst = adj.coalesce().indices()

# Create the heterograph
g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
g = g.int().to(device)

# convert idx_train, idx_val, idx_test to boolean masks
train_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
val_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
test_mask = torch.zeros(adj.shape[0], dtype=torch.bool)
train_mask[idx_train] = True
val_mask[idx_val] = True
test_mask[idx_test] = True
masks = train_mask, val_mask, test_mask

# normalize features
features_orig = features.clone()
features = feature_norm(features)

features = features.to(device)
labels = labels.to(device)
# if sens is a tensor, convert to numpy array
if isinstance(sens, torch.Tensor):
    sens = sens.cpu().numpy()
# if len(sens.shape) == 1:
#     sens = sens.reshape(1, -1)
# create a np array country with "US" for 1 and "oversea" for 0 in sens
race = np.array(["White" if s == 1 else "Other" for s in sens])
# # get the index of "AGE" in feat_names
# age_idx = feat_names.index("AGE")
# age = features_orig[:, age_idx].cpu().numpy()
# # cut age into '<25', '25-30', '>=30'
# age_group = np.array(["<25" if a < 25 else "25-30" if a < 30 else ">=30" for a in age])
# # merge country and age_group into an array
# sens = np.stack([country, age_group], axis=1).T
sens = np.stack([race], axis=1).T
sens_names = ["Race"]
# create GCN model
in_size = features.shape[1]
out_size = int(sum(labels.unique() != -1))
model = GCN(in_size, 16, out_size).to(device)
print('116')
# load the model
model.load_state_dict(torch.load('gcn3layer_bail.pth'))
model.eval()
with torch.no_grad():
    logits = model(g, features)
def get_embeddings(g, features, model):
    # capture the output of each layer of the model
    layer_outputs = []
    def hook(module, input, output):
        layer_outputs.append(output)
    model.layers[0].register_forward_hook(hook)
    model.layers[1].register_forward_hook(hook)

    # get the output of each layer
    with torch.no_grad():
        logits = model(g, features)

    return layer_outputs
max_hop = 2
print('135')
e = eug.EUG(model, adj, features, sens, sens_names, max_hop, masks, labels, get_embeddings)

app = e.show()

# pn.serve(app, port=45776)
app.servable()