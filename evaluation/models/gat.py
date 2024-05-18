import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn

# class GAT(nn.Module):
#     def __init__(self, in_size, hid_size, out_size, heads):
#         super().__init__()
#         self.gat_layers = nn.ModuleList()
        
#         # Initial layer
#         self.gat_layers.append(
#             dglnn.GATConv(
#                 in_size,
#                 hid_size,
#                 heads[0],
#                 feat_drop=0.6,
#                 attn_drop=0.6,
#                 activation=F.elu,
#             )
#         )
        
#         # Last layer
#         self.gat_layers.append(
#             dglnn.GATConv(
#                 hid_size * heads[0],
#                 out_size * heads[1],  # Multiplied by number of heads for output layer to allow mean aggregation
#                 heads[1],
#                 feat_drop=0.6,
#                 attn_drop=0.6,
#                 activation=None,  # No activation in last layer
#             )
#         )
        
#         # Final MLP
#         self.mlp = nn.Linear(out_size * heads[1], out_size)

#     def forward(self, g, inputs):
#         h = inputs
#         for i, layer in enumerate(self.gat_layers):
#             h = layer(g, h)
#             if i == 1:  # last layer
#                 h = h.mean(1)
#             else:  # other layer(s)
#                 h = h.flatten(1)

#         return self.mlp(h)  # Apply MLP to the final graph embeddings
class GAT(nn.Module):
    def __init__(self, in_size, hid_size, out_size, heads):
        super().__init__()
        self.gat_layers = nn.ModuleList()
        
        # Initial layer
        self.gat_layers.append(
            dglnn.GATConv(
                in_size,
                hid_size,
                heads[0],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=F.elu,
            )
        )
        
        # Last layer
        self.gat_layers.append(
            dglnn.GATConv(
                hid_size * heads[0],
                out_size * heads[1],  # Multiplied by number of heads for output layer to allow mean aggregation
                heads[1],
                feat_drop=0.6,
                attn_drop=0.6,
                activation=None,  # No activation in last layer
            )
        )
        
        # Final MLP
        self.mlp = nn.Linear(out_size * heads[1], out_size)

    def forward(self, g, inputs):
        h = inputs
        self.attn_weights = []
        for i, layer in enumerate(self.gat_layers):
            h, attn = layer(g, h, get_attention=True)
            self.attn_weights.append(attn)
            if i == 1:  # last layer
                h = h.mean(1)
            else:  # other layer(s)
                h = h.flatten(1)

        return self.mlp(h)  # Apply MLP to the final graph embeddings

    

def get_embeddings(g, features, model):
    # capture the output of each layer of the model
    layer_outputs = []
    
    # Define a hook function that will save the output of each layer
    def hook(module, input, output):
        layer_outputs.append(output[0].view(output[0].shape[0], -1).detach())  # Use detach() to avoid saving the computation graph

    # Register the hook for each layer in the GCN model
    hooks = []
    for layer in model.gat_layers:
        hook_handle = layer.register_forward_hook(hook)
        hooks.append(hook_handle)

    # Perform a forward pass to trigger the hooks and get the outputs
    with torch.no_grad():
        _ = model(g, features)  # We do not need to store the logits here, just need to pass through the model

    # Remove hooks after use to free memory and avoid side effects
    for hook in hooks:
        hook.remove()

    return layer_outputs