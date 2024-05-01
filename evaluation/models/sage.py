import dgl
import dgl.nn as dglnn

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import AddSelfLoop
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, hid_size, "gcn"))
        self.mlp = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return self.mlp(h)
    

def get_embeddings(g, features, model):
    # capture the output of each layer of the model
    layer_outputs = []
    
    # Define a hook function that will save the output of each layer
    def hook(module, input, output):
        layer_outputs.append(output.detach())  # Use detach() to avoid saving the computation graph

    # Register the hook for each layer in the GCN model
    hooks = []
    for layer in model.layers:
        hook_handle = layer.register_forward_hook(hook)
        hooks.append(hook_handle)

    # Perform a forward pass to trigger the hooks and get the outputs
    with torch.no_grad():
        _ = model(g, features)  # We do not need to store the logits here, just need to pass through the model

    # Remove hooks after use to free memory and avoid side effects
    for hook in hooks:
        hook.remove()

    return layer_outputs