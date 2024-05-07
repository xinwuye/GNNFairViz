import torch
import torch.nn as nn
from torch.nn import init
import torch.nn.functional as F

import numpy as np

class GraphConv(nn.Module):
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
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            if add_self:
                self.self_weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
            if att:
                self.att_weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
            if add_self:
                self.self_weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim).cuda())
            if att:
                self.att_weight = nn.Parameter(torch.FloatTensor(input_dim, input_dim).cuda())
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
            # print('in GraphConv forward')
            # print_gpu_memory_stats()
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

class GcnEncoderGraph(nn.Module):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, add_self=False, args=None,):
        super(GcnEncoderGraph, self).__init__()
        self.concat = concat
        add_self = add_self
        if args.method=='GIN':
            add_self=True
        self.bn = bn
        self.num_layers = num_layers
        self.num_aggs = 1

        self.bias = True
        self.gpu = args.gpu
        if args.method == "GAT":
            self.att = True
        else:
            self.att = False
        if args is not None:
            self.bias = args.bias

        self.conv_first, self.conv_block, self.conv_last = self.build_conv_layers(input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=True, dropout=dropout,)
        self.act = nn.ReLU()
        self.label_dim = label_dim

        if concat:
            self.pred_input_dim = hidden_dim * (num_layers - 1) + embedding_dim
        else:
            self.pred_input_dim = embedding_dim
        self.pred_model = self.build_pred_layers(self.pred_input_dim, pred_hidden_dims, label_dim, num_aggs=self.num_aggs)

        for m in self.modules():
            if isinstance(m, GraphConv):
                init.xavier_uniform_(m.weight.data, gain=nn.init.calculate_gain("relu"))
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
            # print('in GcnEncoderGraph gcn_forward')
            # print_gpu_memory_stats()
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

class GNN(GcnEncoderGraph):
    def __init__(self, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims=[], concat=True, bn=True, dropout=0.0, args=None,):
        super(GNN, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims, concat, bn, dropout, args=args,)
        if hasattr(args, "loss_weight"):
            print("Loss weight: ", args.loss_weight)
            self.celoss = nn.CrossEntropyLoss(weight=args.loss_weight)
        else:
            self.celoss = nn.CrossEntropyLoss()

    def forward(self, x, adj, batch_num_nodes=None, **kwargs):
        # mask
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        self.adj_atts = []
        # print('in GNN forward')
        # print_gpu_memory_stats()
        self.embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask)
        pred = self.pred_model(self.embedding_tensor)
        return pred, 0

    def loss(self, pred, label):
        pred = torch.transpose(pred, 1, 2)
        return self.celoss(pred, label)

class SoftPoolingGcnEncoder(GcnEncoderGraph):
    def __init__(self, max_num_nodes, input_dim, hidden_dim, embedding_dim, label_dim, num_layers, assign_hidden_dim, assign_ratio=0.25, assign_num_layers=-1, num_pooling=1, pred_hidden_dims=[50], concat=True, bn=True, dropout=0.0, linkpred=True, assign_input_dim=-1, args=None,):

        super(SoftPoolingGcnEncoder, self).__init__(input_dim, hidden_dim, embedding_dim, label_dim, num_layers, pred_hidden_dims=pred_hidden_dims, concat=concat, args=args,)
        add_self = not concat
        self.num_pooling = num_pooling
        self.linkpred = linkpred
        self.assign_ent = True

        self.conv_first_after_pool = []
        self.conv_block_after_pool = []
        self.conv_last_after_pool = []
        for i in range(num_pooling):
            self.conv_first2, self.conv_block2, self.conv_last2 = self.build_conv_layers(self.pred_input_dim, hidden_dim, embedding_dim, num_layers, add_self, normalize=True, dropout=dropout,)
            self.conv_first_after_pool.append(self.conv_first2)
            self.conv_block_after_pool.append(self.conv_block2)
            self.conv_last_after_pool.append(self.conv_last2)

        assign_dims = []
        if assign_num_layers == -1:
            assign_num_layers = num_layers
        if assign_input_dim == -1:
            assign_input_dim = input_dim

        self.assign_conv_first_modules = []
        self.assign_conv_block_modules = []
        self.assign_conv_last_modules = []
        self.assign_pred_modules = []
        assign_dim = int(max_num_nodes * assign_ratio)
        for i in range(num_pooling):
            assign_dims.append(assign_dim)
            self.assign_conv_first, self.assign_conv_block, self.assign_conv_last = self.build_conv_layers(assign_input_dim, assign_hidden_dim, assign_dim, assign_num_layers, add_self, normalize=True,)
            assign_pred_input_dim = (assign_hidden_dim * (num_layers - 1) + assign_dim if concat else assign_dim)
            self.assign_pred = self.build_pred_layers(assign_pred_input_dim, [], assign_dim, num_aggs=1)

            # next pooling layer
            assign_input_dim = embedding_dim
            assign_dim = int(assign_dim * assign_ratio)

            self.assign_conv_first_modules.append(self.assign_conv_first)
            self.assign_conv_block_modules.append(self.assign_conv_block)
            self.assign_conv_last_modules.append(self.assign_conv_last)
            self.assign_pred_modules.append(self.assign_pred)

        self.pred_model = self.build_pred_layers(self.pred_input_dim * (num_pooling + 1), pred_hidden_dims, label_dim, num_aggs=self.num_aggs,)

        for m in self.modules():
            if isinstance(m, GraphConv):
                m.weight.data = init.xavier_uniform(m.weight.data, gain=nn.init.calculate_gain("relu"))
                if m.bias is not None:
                    m.bias.data = init.constant(m.bias.data, 0.0)

    def forward(self, x, adj, batch_num_nodes, **kwargs):
        if "assign_x" in kwargs:
            x_a = kwargs["assign_x"]
        else:
            x_a = x
        max_num_nodes = adj.size()[1]
        if batch_num_nodes is not None:
            embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
        else:
            embedding_mask = None

        out_all = []
        embedding_tensor = self.gcn_forward(x, adj, self.conv_first, self.conv_block, self.conv_last, embedding_mask)

        out, _ = torch.max(embedding_tensor, dim=1)
        out_all.append(out)
        if self.num_aggs == 2:
            out = torch.sum(embedding_tensor, dim=1)
            out_all.append(out)

        for i in range(self.num_pooling):
            if batch_num_nodes is not None and i == 0:
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
            else:
                embedding_mask = None

            self.assign_tensor = self.gcn_forward(x_a, adj, self.assign_conv_first_modules[i], self.assign_conv_block_modules[i], self.assign_conv_last_modules[i], embedding_mask,)
            self.assign_tensor = nn.Softmax(dim=-1)(self.assign_pred(self.assign_tensor))
            if embedding_mask is not None:
                self.assign_tensor = self.assign_tensor * embedding_mask

            x = torch.matmul(torch.transpose(self.assign_tensor, 1, 2), embedding_tensor)
            adj = torch.transpose(self.assign_tensor, 1, 2) @ adj @ self.assign_tensor
            x_a = x

            embedding_tensor = self.gcn_forward(x, adj, self.conv_first_after_pool[i], self.conv_block_after_pool[i], self.conv_last_after_pool[i],)

            out, _ = torch.max(embedding_tensor, dim=1)
            out_all.append(out)
            if self.num_aggs == 2:
                out = torch.sum(embedding_tensor, dim=1)
                out_all.append(out)

        if self.concat:
            output = torch.cat(out_all, dim=1)
        else:
            output = out
        ypred = self.pred_model(output)
        return ypred

    def loss(self, pred, label, adj=None, batch_num_nodes=None, adj_hop=1):
        eps = 1e-7
        loss = super(SoftPoolingGcnEncoder, self).loss(pred, label)
        if self.linkpred:
            max_num_nodes = adj.size()[1]
            pred_adj0 = self.assign_tensor @ torch.transpose(self.assign_tensor, 1, 2)
            tmp = pred_adj0
            pred_adj = pred_adj0
            for adj_pow in range(adj_hop - 1):
                tmp = tmp @ pred_adj0
                pred_adj = pred_adj + tmp
            pred_adj = torch.min(pred_adj, torch.Tensor(1).cuda())
            self.link_loss = -adj * torch.log(pred_adj + eps) - (1 - adj) * torch.log(1 - pred_adj + eps)
            if batch_num_nodes is None:
                num_entries = max_num_nodes * max_num_nodes * adj.size()[0]
                print("Warning: calculating link pred loss without masking")
            else:
                num_entries = np.sum(batch_num_nodes * batch_num_nodes)
                embedding_mask = self.construct_mask(max_num_nodes, batch_num_nodes)
                adj_mask = embedding_mask @ torch.transpose(embedding_mask, 1, 2)
                self.link_loss[1 - adj_mask.byte()] = 0.0

            self.link_loss = torch.sum(self.link_loss) / float(num_entries)
            return loss + self.link_loss
        return loss

def print_gpu_memory_stats(device_id=0):
    t = torch.cuda.get_device_properties(device_id).total_memory
    r = torch.cuda.memory_reserved(device_id)
    a = torch.cuda.memory_allocated(device_id)
    f = r-a  # free inside reserved

    print(f"Total GPU Memory: {t / (1024**3):.2f} GB")
    print(f"Reserved Memory: {r / (1024**3):.2f} GB")
    print(f"Allocated Memory: {a / (1024**3):.2f} GB")
    print(f"Free Memory: {f / (1024**3):.2f} GB")