# import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# from dgl.data import GINDataset
# from dgl.dataloading import GraphDataLoader
from dgl.nn.pytorch.conv import GINConv
from dgl.nn.pytorch.glob import SumPooling, AvgPooling
# from sklearn.model_selection import StratifiedKFold
# from torch.utils.data.sampler import SubsetRandomSampler


class MLP(nn.Module):
    """Construct two-layer MLP-type aggreator for GIN model"""

    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.linears = nn.ModuleList()
        # two-layer MLP
        self.linears.append(nn.Linear(input_dim, hidden_dim, bias=False))
        self.linears.append(nn.Linear(hidden_dim, output_dim, bias=False))
        self.batch_norm = nn.BatchNorm1d((hidden_dim))

    def forward(self, x):
        h = x
        h = F.relu(self.batch_norm(self.linears[0](h)))
        return self.linears[1](h)


class GIN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=5):
        super().__init__()
        self.ginlayers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Initializing GIN layers with MLP aggregators
        for layer in range(num_layers):
            if layer == 0:
                mlp = MLP(input_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(hidden_dim, hidden_dim, hidden_dim)
            self.ginlayers.append(GINConv(mlp, learn_eps=False))  # Epsilon not learned
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Output layer to map the final node features to class scores
        self.node_classifier = nn.Linear(hidden_dim, output_dim)
        self.drop = nn.Dropout(0.5)

    def forward(self, g, h):
        for i, layer in enumerate(self.ginlayers):
            h = layer(g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
        
        # Apply dropout and classify nodes
        h = self.drop(h)
        node_scores = self.node_classifier(h)
        return node_scores


def get_embeddings(g, features, model):
    # capture the output of each layer of the model
    layer_outputs = []
    
    # Define a hook function that will save the output of each layer
    def hook(module, input, output):
        layer_outputs.append(output.detach())  # Use detach() to avoid saving the computation graph

    # Register the hook for each layer in the GCN model
    hooks = []
    for layer in model.ginlayers:
        hook_handle = layer.register_forward_hook(hook)
        hooks.append(hook_handle)

    # Perform a forward pass to trigger the hooks and get the outputs
    with torch.no_grad():
        _ = model(g, features)  # We do not need to store the logits here, just need to pass through the model

    # Remove hooks after use to free memory and avoid side effects
    for hook in hooks:
        hook.remove()

    return layer_outputs