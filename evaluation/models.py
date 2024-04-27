import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.nn as dglnn


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