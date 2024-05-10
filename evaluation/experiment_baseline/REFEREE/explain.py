import math
import time
import os
from scipy.stats import wasserstein_distance
import matplotlib
import matplotlib.colors as colors
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import random
import networkx as nx
import numpy as np
import pandas as pd
# import seaborn as sns
import util
# import tensorboardX.utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN
import dgl

import pdb
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor

def sparse_softmax(v):
    # This function will apply softmax only on non-zero elements of the sparse tensor.
    # Note: This is an approximation and changes the meaning of softmax fundamentally.
    max_val = torch.max(v._values())
    exps = torch.exp(v._values() - max_val)
    sum_of_exps = torch.sum(exps)
    softmax_values = exps / sum_of_exps
    return torch.sparse_coo_tensor(v._indices(), softmax_values, v.size())

def sparse_kl_div(log_prob_sparse, prob_sparse):
    # Direct computation of KL divergence on non-zero entries
    kl_div_values = prob_sparse._values() * (torch.log(prob_sparse._values()) - log_prob_sparse._values())
    return torch.sum(kl_div_values)

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True)
        norms_2 = torch.sum(sample_2**2, dim=1, keepdim=True)
        norms = (norms_1.expand(n_1, n_2) +
                 norms_2.transpose(0, 1).expand(n_1, n_2))
        distances_squared = norms - 2 * sample_1.mm(sample_2.t())
        return torch.sqrt(eps + torch.abs(distances_squared))
    else:
        dim = sample_1.size(1)
        expanded_1 = sample_1.unsqueeze(1).expand(n_1, n_2, dim)
        expanded_2 = sample_2.unsqueeze(0).expand(n_1, n_2, dim)
        differences = torch.abs(expanded_1 - expanded_2) ** norm
        inner = torch.sum(differences, dim=2, keepdim=False)
        return (eps + inner) ** (1. / norm)

def wasserstein(x, y, p=0.5, lam=10, its=10, cuda=False):
    if len(x.shape)<2:
        x=x.unsqueeze(0)
    if len(y.shape)<2:
        y=y.unsqueeze(0)
    nx = x.shape[0]
    ny = y.shape[0]
    M = pdist(x, y)
    M_mean = torch.mean(M)
    M_drop = F.dropout(M, 0.0 / (nx * ny))
    delta = torch.max(M_drop).cpu().detach()
    eff_lam = (lam / M_mean).cpu().detach()

    row = delta * torch.ones(M[0:1, :].shape)
    col = torch.cat([delta * torch.ones(M[:, 0:1].shape), torch.zeros((1, 1))], 0)
    if cuda:
        row = row.cuda()
        col = col.cuda()
    Mt = torch.cat([M, row], 0)
    Mt = torch.cat([Mt, col], 1)
    a = torch.cat([p * torch.ones((nx, 1)) / nx, (1 - p) * torch.ones((1, 1))], 0)
    b = torch.cat([(1 - p) * torch.ones((ny, 1)) / ny, p * torch.ones((1, 1))], 0)
    Mlam = eff_lam * Mt
    temp_term = torch.ones(1) * 1e-6
    if cuda:
        temp_term = temp_term.cuda()
        a = a.cuda()
        b = b.cuda()
    K = torch.exp(-Mlam) + temp_term
    ainvK = K / a
    u = a
    for i in range(its):
        u = 1.0 / (ainvK.matmul(b / torch.t(torch.t(u).matmul(K))))
        if cuda:
            u = u.cuda()
    v = b / (torch.t(torch.t(u).matmul(K)))
    if cuda:
        v = v.cuda()
    upper_t = u * (torch.t(v) * K).detach()
    E = upper_t * Mt
    D = 2 * torch.sum(E)
    if cuda:
        D = D.cuda()
    return D, Mlam

class REFEREE:
    def __init__(self, model, adj, feat, label, pred, train_idx, sens, args):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat.cpu().detach().numpy()
        self.label = label.cpu().detach()
        self.pred = pred.cpu().detach().numpy()
        self.train_idx = train_idx
        self.sens = sens

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # check if adj is pytorch sparse tensor type
        if adj.is_sparse:
            src, dst = adj.coalesce().indices()
            # Create the heterograph
            g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
            g = g.int().to(device)

        self.g = g

        if args.dataset=='credit':
            self.test_idx=[one for one in list(range(self.feat.shape[0])) if one not in self.train_idx]
        elif args.dataset=='german':
            np.random.seed(0)
            self.test_idx=np.random.choice(list(range(self.feat.shape[0])),size=500 if args.baseline else 200) #200 #100
        else:
            np.random.seed(0)
            self.test_idx=np.random.choice(list(range(self.feat.shape[0])),size=1000 if args.baseline else 500)

        self.node_idx=[]

        self.n_hops = args.num_gc_layers
        # self.neighborhoods = util.neighborhoods(adj=self.adj, n_hops=self.n_hops, use_cuda=use_cuda)
        self.args = args
        self.print_training = args.debug_mode

        self.explain_true=[]
        self.explain_true_wass=[]
        self.explain_true_minuswass=[]
        self.wass_dis=[]
        self.wass_dis_ori=[]
        self.wass_dis_att=[]
        self.start_wass_dis=[]
        self.start_wass_dis_ori=[]
        self.start_wass_dis_att=[]
        self.start_wass_dis_unfair=[]
        self.wass_dis_unfair=[]

        self.folder_name=self.args.dataset+'_'+self.args.explainer_backbone+'_'+self.args.method
        os.makedirs('log/'+self.folder_name,exist_ok=True)

        self.tensor_pred=torch.tensor(self.pred).cuda().softmax(-1)

    def explain(self, node_idx):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        seed = 100


        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

        node_idx_new, sub_adj, sub_feat, sub_label, neighbors = self.extract_neighborhood(node_idx, max_size=10 if self.args.dataset=='bail' else None)
        # sub_label = np.expand_dims(sub_label, axis=0)

        # sub_adj = np.expand_dims(sub_adj, axis=0)
        # sub_feat = np.expand_dims(sub_feat, axis=0)

        # adj   = torch.tensor(sub_adj, dtype=torch.float)
        # x     = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).cuda()
        # label = torch.tensor(sub_label, dtype=torch.long)
        adj = sub_adj
        x = torch.tensor(sub_feat, requires_grad=True, dtype=torch.float).cuda()
        label = torch.tensor(sub_label, dtype=torch.long)

        pred_label = np.argmax(self.pred[neighbors], axis=1)

        sens = self.sens

        explainer_fair = ExplainModule(adj=adj, x=x, model=self.model, label=label, args=self.args)
        if self.args.gpu:
            explainer_fair = explainer_fair.cuda()


        explainer_unfair = ExplainModule(adj=adj, x=x, model=self.model, label=label, args=self.args)
        if self.args.gpu:
            explainer_unfair = explainer_unfair.cuda()

        self.model.eval()

        for epoch in range(self.args.num_epochs):

            explainer_unfair.optimizer.zero_grad()
            # _, _ = explainer_unfair(node_idx_new)
            explainer_unfair(node_idx_new)
            ypred_unfair = explainer_unfair.cal_WD_ypred(node_idx_new, threshold=self.args.threshold)
            loss = explainer_unfair.loss(ypred_unfair, pred_label, node_idx_new) * self.args.fidelity_fair_weight

            # # index=self.feat[self.test_idx][:,-1]==1-self.feat[node_idx][-1]
            # index = self.sens[self.test_idx] != self.sens[node_idx]
            # new_tensor_pred=self.tensor_pred.clone()
            # new_tensor_pred[node_idx]=ypred_unfair
            # # new_index=self.feat[self.test_idx][:,-1]==self.feat[node_idx][-1]
            # new_index = self.sens[self.test_idx] == self.sens[node_idx]
            # wass_dis=wasserstein(self.tensor_pred[self.test_idx][index],new_tensor_pred[self.test_idx][new_index],cuda=True)[0]

            wass_dis = 0
            # get unique values of sens
            # sens_unique = torch.unique(sens)
            sens_unique = list(set(sens))
            # loop over evey pair of values in sens_unique
            for i in range(len(sens_unique)):
                for j in range(i+1, len(sens_unique)):
                    # get the indices of the nodes with the current pair of values
                    index1 = sens == sens_unique[i]
                    index2 = sens == sens_unique[j]
                    # convert to torch tensor
                    index1 = torch.tensor(index1, dtype=torch.bool, device=device)
                    index2 = torch.tensor(index2, dtype=torch.bool, device=device)
                    # get the predictions of the nodes with the current pair of values
                    pred1 = self.tensor_pred[index1]
                    pred2 = self.tensor_pred[index2]
                    # compute the Wasserstein distance between the predictions of the nodes with the current pair of values
                    wass_dis = wass_dis + wasserstein(pred1, pred2, cuda=True)[0]

            loss -= wass_dis*self.args.WD_fair_weight

            loss.backward(retain_graph=True)
            explainer_unfair.optimizer.step()

            ###############################################
            explainer_fair.optimizer.zero_grad()

            # _, _ = explainer_unfair(node_idx_new)
            # _, _ = explainer_fair(node_idx_new)
            explainer_unfair(node_idx_new)
            explainer_fair(node_idx_new)

            ypred = explainer_fair.cal_WD_ypred(node_idx_new, threshold=self.args.threshold)

            loss = explainer_fair.loss(ypred, pred_label, node_idx_new)* self.args.fidelity_fair_weight

            # # index=self.feat[self.test_idx][:,-1]==1-self.feat[node_idx][-1]
            # index = self.sens[self.test_idx] != self.sens[node_idx]
            # new_tensor_pred=self.tensor_pred.clone()
            # new_tensor_pred[node_idx]=ypred
            # # new_index=self.feat[self.test_idx][:,-1]==self.feat[node_idx][-1]
            # new_index = self.sens[self.test_idx] == self.sens[node_idx]
            # wass_dis=wasserstein(self.tensor_pred[self.test_idx][index],new_tensor_pred[self.test_idx][new_index],cuda=True)[0]

            wass_dis = 0
            # get unique values of sens
            # sens_unique = torch.unique(sens)
            sens_unique = list(set(sens))
            # loop over evey pair of values in sens_unique
            for i in range(len(sens_unique)):
                for j in range(i+1, len(sens_unique)):
                    # get the indices of the nodes with the current pair of values
                    index1 = sens == sens_unique[i]
                    index2 = sens == sens_unique[j]
                    # get the predictions of the nodes with the current pair of values
                    pred1 = self.tensor_pred[index1]
                    pred2 = self.tensor_pred[index2]
                    # compute the Wasserstein distance between the predictions of the nodes with the current pair of values
                    wass_dis = wass_dis + wasserstein(pred1, pred2, cuda=True)[0]

            loss += wass_dis*self.args.WD_fair_weight

            # KL_loss=(0.5*F.kl_div(torch.log(explainer_fair.masked_adj.flatten().softmax(-1)),explainer_unfair.masked_adj.flatten().softmax(-1))*+0.5*F.kl_div(torch.log(explainer_unfair.masked_adj.flatten().softmax(-1)),explainer_fair.masked_adj.flatten().softmax(-1)))
            softmax_fair = sparse_softmax(explainer_fair.masked_adj)
            softmax_unfair = sparse_softmax(explainer_unfair.masked_adj)
            # Applying KL divergence
            log_softmax_fair = torch.sparse_coo_tensor(softmax_fair._indices(), torch.log(softmax_fair._values()), softmax_fair.size())
            log_softmax_unfair = torch.sparse_coo_tensor(softmax_unfair._indices(), torch.log(softmax_unfair._values()), softmax_unfair.size())
            KL_loss = 0.5 * sparse_kl_div(log_softmax_fair, softmax_unfair) + 0.5 * sparse_kl_div(log_softmax_unfair, softmax_fair)

            loss-=KL_loss*self.args.KL_weight
            loss.backward(retain_graph=True)

            explainer_fair.optimizer.step()

        # masked_adj_fair = (explainer_fair.masked_adj.cpu().detach().numpy() * sub_adj.squeeze())
        # masked_adj_unfair = (explainer_unfair.masked_adj.cpu().detach().numpy() * sub_adj.squeeze())

        # self.node_idx.append(node_idx)

        # if self.args.dataset=='german':
        #     fname = 'neighbors_' + 'node_idx_'+str(node_idx)+'.npy'
        #     with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
        #         np.save(outfile, np.asarray(neighbors.tolist()+[node_idx_new]))

        #     fname = 'masked_adj_' + 'node_idx_'+str(node_idx)+'.npy'
        #     with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
        #         np.save(outfile, np.asarray([masked_adj_fair.copy(),masked_adj_unfair.copy()]))

        masked_adj_unfair = explainer_unfair.masked_adj.coalesce()
        values = masked_adj_unfair.values().detach()
        indices = masked_adj_unfair.indices()
        edge_mask = values > 0
        values = values[edge_mask]
        indices = indices[:, edge_mask]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        indices_all = torch.tensor(neighbors, device=device)[indices]
        
        feat_mask = torch.sigmoid(explainer_unfair.feat_mask.detach())
        
        return values, indices_all, feat_mask, torch.tensor(neighbors, device=device)

    def explain_nodes_gnn_stats(self, node_indices):
        results = []
        for node_idx in tqdm(node_indices):
            if self.args.dataset=='credit':
                values, indices_all, feat_mask = self.explain(self.test_idx[node_idx])
            else:
                values, indices_all, feat_mask = self.explain(node_idx)
            results.append((values, indices_all, feat_mask))
        return results
        

    # def extract_neighborhood(self, node_idx, max_size=None):
    #     # neighbors_adj_row = self.neighborhoods[0][node_idx, :]
    #     neighbors_adj_row = self.neighborhoods[None, ...][0][node_idx, :]
    #     node_idx_new = sum(neighbors_adj_row[:node_idx])
    #     neighbors = np.nonzero(neighbors_adj_row)[0]

    #     if max_size!=None:
    #         neighbors=neighbors.tolist()
    #         neighbors.remove(node_idx)
    #         neighbors=neighbors[:max_size]
    #         neighbors.append(node_idx)
    #         node_idx_new=len(neighbors)-1

    #     # sub_adj = self.adj[0][neighbors][:, neighbors]
    #     # sub_adj = self.adj[None, ...][0][neighbors][:, neighbors]
    #     neighbors_tensor = torch.tensor(neighbors, dtype=torch.long)
    #     sub_adj = torch.index_select(self.adj, 0, neighbors_tensor)
    #     sub_adj = torch.index_select(sub_adj, 1, neighbors_tensor)
    #     sub_feat = self.feat[neighbors]
    #     sub_label = self.label[neighbors]
    #     return node_idx_new, sub_adj, sub_feat, sub_label, neighbors
    def extract_neighborhood(self, node_idx, max_size=None):
        neighbors_adj_row = self.adj[node_idx].to_dense().unsqueeze(0)
        for i in range(self.n_hops - 1):
            neighbors_adj_row += torch.sparse.mm(neighbors_adj_row, self.adj)
        # print((neighbors_adj_row != 0)[node_idx])
        node_idx_new = int((neighbors_adj_row.squeeze() != 0)[:node_idx].sum().item())
        neighbors = np.nonzero(neighbors_adj_row.squeeze()).squeeze()

        # sub_adj = self.adj[0][neighbors][:, neighbors]
        # sub_adj = self.adj[None, ...][0][neighbors][:, neighbors]
        neighbors_tensor = torch.tensor(neighbors, dtype=torch.long)
        sub_adj = torch.index_select(self.adj, 0, neighbors_tensor)
        sub_adj = torch.index_select(sub_adj, 1, neighbors_tensor)
        sub_feat = self.feat[neighbors]
        sub_label = self.label[neighbors]
        return node_idx_new, sub_adj, sub_feat, sub_label, neighbors

class ExplainModule(nn.Module):
    def __init__(self, adj, x, model, label, args, use_sigmoid=True):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.label = label
        self.args = args
        self.use_sigmoid = use_sigmoid
        num_nodes = adj.size()[1]
        self.PGE_module=nn.Sequential(nn.Linear(self.x.shape[-1], num_nodes),).cuda()
        self.mask = self.construct_edge_mask(num_nodes)
        self.feat_mask = self.construct_feat_mask()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # check if adj is pytorch sparse tensor type
        if adj.is_sparse:
            src, dst = adj.coalesce().indices()
            # Create the heterograph
            g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
            g = g.int().to(device)

        self.g = g

        params = [self.mask, self.feat_mask]
        if args.explainer_backbone!='GNNExplainer':
            params.extend(self.PGE_module.parameters())

        self.optimizer = util.build_optimizer(args, params)

    def construct_edge_mask(self, num_nodes):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (num_nodes + num_nodes))
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask
    
    def construct_feat_mask(self):
        mask = nn.Parameter(torch.FloatTensor(self.x.shape[-1]))
        std = nn.init.calculate_gain("relu") * math.sqrt(1.0 / self.x.shape[-1])
        with torch.no_grad():
            mask.normal_(1.0, std)
        return mask

    def _masked_adj(self):
        if self.args.explainer_backbone=='PGExplainer':
            sym_mask = self.PGE_module(self.x).squeeze(0)
        else:
            sym_mask = self.mask

        sym_mask = torch.sigmoid(sym_mask)
        sym_mask = (sym_mask + sym_mask.t()) / 2
        adj = self.adj.cuda() if self.args.gpu else self.adj

        self.sym_mask=sym_mask
        masked_adj = adj * sym_mask
        return masked_adj

    def mask_density(self):
        mask_sum = torch.sum(self._masked_adj()).cpu()
        adj_sum = torch.sum(self.adj)
        return mask_sum / (adj_sum+1e-9)

    def forward(self, node_idx):
        # x = self.x.cuda() if self.args.gpu else self.x
        self.masked_adj = self._masked_adj()
        # print('shape:',x.shape, self.masked_adj.shape)
        # ypred, adj_att = self.model(x, self.masked_adj)
        # node_pred = ypred[node_idx, :]
        # res = nn.Softmax(dim=0)(node_pred)
        # return res, adj_att
    
    def cal_WD_ypred(self, node_idx,threshold):
        x = self.x.cuda() if self.args.gpu else self.x
        feat_mask = torch.sigmoid(self.feat_mask)
        x = x * feat_mask
        self.masked_adj = self._masked_adj()

        ori_mask = self.sym_mask  # * self.adj.cuda()  #modified
        ranking = ori_mask.flatten()
        ranking, _=torch.sort(ranking)


        if threshold < len(ranking):  #modified
            threshold_value = ranking[-min(threshold, len(ranking))]
            #threshold_value=0
            #print(threshold_value)
            #print(torch.where(ori_mask>threshold_value, ori_mask, 0))

            #print(self.adj.shape)
            #print(self.adj.sum())
            self.masked_adj = self.adj.cuda() * torch.where(ori_mask>threshold_value, ori_mask, torch.tensor(0.0).cuda())



        new_adj = self.masked_adj

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # check if adj is pytorch sparse tensor type
        if new_adj.is_sparse:
            src, dst = new_adj.coalesce().indices()
            # Create the heterograph
            g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
            g = g.int().to(device)

        ypred = self.model(g, x)
        node_pred = ypred[node_idx, :]
        res = nn.Softmax(dim=0)(node_pred)

        return res

    # def adj_feat_grad(self, node_idx, pred_label_node):
    #     self.model.zero_grad()
    #     self.adj.requires_grad = True
    #     self.x.requires_grad = True
    #     if self.adj.grad is not None:
    #         self.adj.grad.zero_()
    #         self.x.grad.zero_()
    #     if self.args.gpu:
    #         adj = self.adj.cuda()
    #         x = self.x.cuda()
    #     else:
    #         x, adj = self.x, self.adj

    #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #     # check if adj is pytorch sparse tensor type
    #     if adj.is_sparse:
    #         src, dst = adj.coalesce().indices()
    #         # Create the heterograph
    #         g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    #         g = g.int().to(device)        

    #     # ypred = self.model(x, adj)
    #     ypred = self.model(g, x)
    #     logit = nn.Softmax(dim=0)(ypred[node_idx, :])
    #     logit = logit[pred_label_node]
    #     loss = -torch.log(logit)
    #     loss.backward()
    #     return self.adj.grad, self.x.grad

    def loss(self, pred, pred_label, node_idx):
        pred_label_node = pred_label[node_idx]
        logit=pred[pred_label_node]
        pred_loss = -torch.log(logit)
        mask = torch.sigmoid(self.mask)

        size_loss = nn.ReLU()(torch.sum(mask)-self.args.threshold)*self.args.size_weight

        loss = pred_loss + size_loss

        return loss


