import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

class GCN(nn.Module):
    def __init__(self, in_size, hid_size, out_size, num_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        assert num_layers >= 2, "Number of layers should be at least 2."
        # First layer
        self.layers.append(dglnn.GraphConv(in_size, hid_size, activation=F.relu))
        # Hidden layers
        for _ in range(1, num_layers-1):
            self.layers.append(dglnn.GraphConv(hid_size, hid_size, activation=F.relu))
        # Last layer
        self.layers.append(dglnn.GraphConv(hid_size, hid_size))
        # Output layer replaced by a simple MLP (single linear transformation)
        self.mlp = nn.Linear(hid_size, out_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, g, features):
        h = features
        for layer in self.layers:
            h = layer(g, h)
            h = self.dropout(h)
        return self.mlp(h)  # Apply MLP to the final graph embeddings

    

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