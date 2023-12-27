import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

class GraphConv(nn.Module): # used in GNN.
    def __init__(self, input_dim, output_dim, add_self=False, normalize_embedding=False, dropout=0.0, bias=True, gpu=True, att=False):
        super(GraphConv, self).__init__()
        self.att = att
        self.add_self = add_self
        self.dropout = dropout
        if dropout > 0.001:
            self.dropout_layer = nn.Dropout(p=dropout)
        self.normalize_embedding = normalize_embedding
        self.input_dim = input_dim
        self.output_dim = output_dim
        if not gpu:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim)) # a layer in PyTorch that applies dropout to the input tensor.
            if add_self:
                self.self_weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            if att:
                self.att_weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        else:
            # self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
            # change the above line to the following line to use the CPU
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            if add_self:
                # self.self_weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
                # change the above line to the following line to use the CPU
                self.self_weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            if att:
                # self.att_weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim).cuda())
                # change the above line to the following line to use the CPU
                self.att_weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        if bias:
            if not gpu:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim))
            else:
                self.bias = nn.Parameter(torch.FloatTensor(output_dim).cuda())
        else:
            self.bias = None
    def forward(self, x, adj):
        if self.dropout > 0.001:
            x = self.dropout_layer(x)
        if self.att:
            x_att = torch.matmul(x, self.att_weight)
            att = x_att @ x_att.permute(0, 2, 1)
            adj = adj * att
        y = torch.matmul(adj, x)
        y = torch.matmul(y, self.weight)
        if self.add_self:
            self_emb = torch.matmul(x, self.self_weight)
            y += self_emb
        if self.bias is not None:
            y = y + self.bias
        if self.normalize_embedding:
            y = F.normalize(y, p=2, dim=2)
        return y, adj

class GcnEncoderGraph(nn.Module): # used in GNN.
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, add_self=False, args=None,):
        super(GcnEncoderGraph, self).__init__() # calling the init function of an obj of the parent class, the same as super().__init__()
        self.concat = concat
        add_self = add_self
        if args.method=='GIN':
            add_self=True
        self.bn = bn
        self.num_layers = num_layers # num_layers == args.num_gc_layers in configs.py by default
        self.num_aggs = 1

        self.bias = True
        self.gpu = args.gpu
        if args.method == "GAT":
            self.att = True
        else:
            self.att = False
        if args is not None:
            self.bias = args.bias
        # conv_fist is the first layer of the GNN, conv_block is the middle layers of the GNN (3-2=1 layer only in this class), conv_last is the last layer of the GNN
        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=True, dropout=dropout,)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs)

        for m in self.modules(): # traverse the entire network. This is often used for tasks like model introspection, weight initialization, and setting different parameters for different modules.
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu")) # initializes the weights of a neural network layer with a uniform distribution scaled by the Xavier factor.
                if m.att:
                    init.xavier_uniform_(m.att_weight.data, gain=nn.init.calculate_gain("relu"))
                if m.add_self:
                    init.xavier_uniform_(m.self_weight.data, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)

    def build_conv_layers(self, input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=False, dropout=0.0,):
        conv_first = GraphConv(input_dim=input_dim, output_dim=hidden_dim, add_self=add_self, normalize_embedding=normalize, bias=self.bias, gpu=self.gpu, att=self.att,)
        conv_block = nn.ModuleList([GraphConv(input_dim=hidden_dim, output_dim=hidden_dim, add_self=add_self, normalize_embedding=normalize, dropout=dropout, bias=self.bias, gpu=self.gpu, att=self.att,) for i in range(num_layers - 2)])
        conv_last = GraphConv(input_dim=hidden_dim, output_dim=embedding_dim, add_self=add_self, normalize_embedding=normalize, bias=self.bias, gpu=self.gpu, att=self.att,)
        return conv_first, conv_block, conv_last

    def build_pred_layers(self, pred_input_dim, pred_hidden_dims, label_dim, num_aggs=1):
        pred_input_dim = pred_input_dim * num_aggs
        if len(pred_hidden_dims) == 0:
            pred_model = nn.Linear(pred_input_dim, label_dim)
        else:
            pred_layers = []
            for pred_dim in pred_hidden_dims:
                pred_layers.append(nn.Linear(pred_input_dim, pred_dim))
                pred_layers.append(self.act)
                pred_input_dim = pred_dim
            pred_layers.append(nn.Linear(pred_dim, label_dim))
            pred_model = nn.Sequential(*pred_layers)
        return pred_model

    def construct_mask(self, max_nodes, batch_num_nodes):
        packed_masks = [torch.ones(int(num)) for num in batch_num_nodes]
        batch_size = len(batch_num_nodes)
        out_tensor = torch.zeros(batch_size, max_nodes)
        for i, mask in enumerate(packed_masks):
            out_tensor[i, : batch_num_nodes[i]] = mask
        return out_tensor.unsqueeze(2).cuda()

    def apply_bn(self, x):
        bn_module = nn.BatchNorm1d(x.size()[1])
        if self.gpu:
            bn_module = bn_module.cuda()
        return bn_module(x)

    def gcn_forward(self, x, adj, conv_first, conv_block, conv_last, embedding_mask=None):

        x, adj_att = conv_first(x, adj)

        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        x_all = [x]
        adj_att_all = [adj_att]
        for i in range(len(conv_block)):
            x, _ = conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            x_all.append(x)
            adj_att_all.append(adj_att)
        x, adj_att = conv_last(x, adj)
        x_all.append(x)
        adj_att_all.append(adj_att)
        x_tensor = torch.cat(x_all, dim=2)
        if embedding_mask is not None:
            x_tensor = x_tensor * embedding_mask
        self.embedding_tensor = x_tensor

        return x_tensor

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            self.embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            self.embedding_mask = None

        x, adj_att = self.conv_first(x, adj)
        x = self.act(x)
        if self.bn:
            x = self.apply_bn(x)
        out_all = []
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        adj_att_all = [adj_att]
        for i in range(self.num_layers - 2):
            x, adj_att = self.conv_block[i](x, adj)
            x = self.act(x)
            if self.bn:
                x = self.apply_bn(x)
            out, _ = torch.max(x, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(x, dim=1)
                out_all.append(out)
            adj_att_all.append(adj_att)
        x, adj_att = self.conv_last(x, adj)
        adj_att_all.append(adj_att)
        # x = self.act(x)
        out, _ = torch.max(x, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(x, dim=1)
            out_all.append(out)
        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out

        adj_att_tensor = torch.stack(adj_att_all, dim=3)

        self.embedding_tensor = output
        ypred = self.pred_model(output)
        return ypred, adj_att_tensor

    def loss(self, pred, label, type="softmax"):
        if type == "softmax":
            return F.cross_entropy(pred, label, size_average=True)
        elif type == "margin":
            batch_size = pred.size()[0]
            label_onehot = torch.zeros(batch_size, self.label_dim).long().cuda()
            label_onehot.scatter_(1, label.view(-1, 1), 1)
            return torch.nn.MultiLabelMarginLoss()(pred, label_onehot)

class GNN(GcnEncoderGraph): # used in default mode
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None,):
        super(GNN, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims, concat, bn, dropout, args=args,)
        if hasattr(args, "loss_weight"):
            print("Loss weight: ", args.loss_weight)
            self.celoss = nn.CrossEntropyLoss(weight=args.loss_weight)
        else:
            self.celoss = nn.CrossEntropyLoss()

    def forward(self, x, adj, batch_num_nodes=None, **kwargs): # When a child DNN class inherits from a parent DNN class in PyTorch, the forward method of the child class will override the forward method of the parent class. When the forward method of the child class is called, only the forward method of the child class will be executed, and the forward method of the parent class will not be called.
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.adj_atts = []
        self.embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        pred = self.pred_model(self.embedding_tensor)
        
        return pred, 0

    def loss(self, pred, label):
        pred = torch.transpose(pred, 1, 2)
        return self.celoss(pred, label)

