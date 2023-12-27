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
# import util
from . import util
# import tensorboardX.utils
import torch.nn.functional as F
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score, recall_score, precision_score, roc_auc_score, precision_recall_curve
from sklearn.cluster import DBSCAN

import pdb
from tqdm import tqdm

use_cuda = torch.cuda.is_available()
FloatTensor = torch.cuda.FloatTensor if use_cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if use_cuda else torch.LongTensor
Tensor = FloatTensor
torch.manual_seed(42)

def pdist(sample_1, sample_2, norm=2, eps=1e-5):
    n_1, n_2 = sample_1.size(0), sample_2.size(0)
    norm = float(norm)
    if norm == 2.:
        norms_1 = torch.sum(sample_1**2, dim=1, keepdim=True) # calculate sum of squares of each row
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

class GlobalExplainer:
    def __init__(self, model, adj, feat, args):
        self.model = model
        self.model.eval()
        self.adj = adj
        self.feat = feat
        self.node_idx=[]
        self.args = args
        self.print_training = args.debug_mode

        self.explain_true=[]
        self.explain_true_wass=[]
        self.explain_true_minuswass=[]
        self.wass_dis=[] # self.wass_dis stores (for every node to be explained) W1(𝑃( ̃Y0), 𝑃( ̃Y1)) after replacing the original predicition of the node to be explained with the prediction of this node obtained by fairness explainer
        self.wass_dis_ori=[]
        self.wass_dis_att=[]
        self.start_wass_dis=[]
        self.start_wass_dis_ori=[] # this (for every node to be explained) the Wasserstein distance between 2 original subgroups' predicted probability distributions, i.e. W1(P(^Y_0), P(^Y_1))
        self.start_wass_dis_att=[]
        self.start_wass_dis_unfair=[]
        self.wass_dis_unfair=[] # self.wass_dis_unfair stores (for every node to be explained) W1(𝑃( ̃Y0), 𝑃( ̃Y1)) after replacing the original predicition of the node to be explained with the prediction of this node obtained by unfairness explainer

        self.folder_name=self.args.dataset+'_'+self.args.explainer_backbone+'_'+self.args.method+'_unfairness_global' # 'german_PGExplainer_GAT in the case of german dataset
        os.makedirs('log/'+self.folder_name,exist_ok=True)

        # self.tensor_pred = model(feat, adj)[0]
        self.tensor_pred = model(feat, adj)[0].flatten(start_dim=0, end_dim=-2)

    def explain(self, sens, sens_name): # explain a single node
        seed = 100
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        adj   = self.adj
        x     = self.feat
        explainer_unfair = ExplainModule(adj=adj, x=x, model=self.model, sens=sens, args=self.args)
        self.model.eval() # sets the model to evaluation mode. In addition to disabling certain modules, calling model.eval() also sets the requires_grad attribute of all parameters to False, which disables autograd. This can make the model's forward pass more efficient, since no gradients need to be computed or stored.

        wass_dis = 0
        # get unique values of sens
        # sens_unique = torch.unique(sens)
        sens_unique = list(set(sens))
        # loop over evey pair of values in sens_unique
        for i in range(len(sens_unique)):
            for j in range(i+1, len(sens_unique)):
                # get the indices of the nodes with the current pair of values
                # index1 = sens == sens_unique[i]
                # index2 = sens == sens_unique[j]
                index1 = list(map(lambda x: x == sens_unique[i], sens))
                index2 = list(map(lambda x: x == sens_unique[j], sens))
                # get the predictions of the nodes with the current pair of values
                # pred1 = self.tensor_pred[0][index1]
                # pred2 = self.tensor_pred[0][index2]
                pred1 = self.tensor_pred[index1]
                pred2 = self.tensor_pred[index2]
                # compute the Wasserstein distance between the predictions of the nodes with the current pair of values
                wass_dis = wass_dis + wasserstein(pred1, pred2)[0]

        self.start_wass_dis_ori.append(wass_dis.cpu().detach().item())

        for epoch in tqdm(range(self.args.num_epochs)):
            explainer_unfair.optimizer.zero_grad() # clear the gradients of all optimized parameters. This function is typically called at the beginning of each training iteration to ensure that the gradients from the previous iteration are not accumulated.
            loss = explainer_unfair.loss()
            loss.backward(retain_graph=True)
            explainer_unfair.optimizer.step()

        masked_adj_unfair = explainer_unfair.masked_adj.detach().cpu().numpy()
        # fname = sens_name + '_masked_adj.npy'
        # with open(os.path.join(self.args.logdir, self.folder_name,fname), 'wb') as outfile:
        #     np.save(outfile, masked_adj_unfair.copy())
        # print('saved masked_adj_unfair to', os.path.join(self.args.logdir, self.folder_name,fname))

        result=pd.DataFrame([{'dataset':self.args.dataset, 
                              'explainer_backbone':self.args.explainer_backbone, 
                              'GNN_type':self.args.method,
                              'Delta B (Promoted)': '{:.4f}'.format(-(wass_dis - (-explainer_unfair.loss())) / wass_dis * 1e4), # positive value with larger absolute value is better
                              'original wass_dis': '{:.4f}'.format(wass_dis),
                              'new_wass_dis': '{:.4f}'.format(-explainer_unfair.loss()),
                              }])

        # with open('./emugge_delta_b.txt', 'a') as f:
        #     f.write(str(float(-(wass_dis - (-explainer_unfair.loss())) / wass_dis * 1e4)) + ',')


        print('Fitting completed.')
        print(result)

        return masked_adj_unfair


class ExplainModule(nn.Module):
    def __init__(self, adj, x, model, sens, args, use_sigmoid=True):
        super(ExplainModule, self).__init__()
        self.adj = adj
        self.x = x
        self.model = model
        self.sens = sens
        self.args = args
        self.use_sigmoid = use_sigmoid
        num_nodes = adj.size()[1]
        # self.PGE_module=nn.Sequential(nn.Linear(self.x.shape[-1], num_nodes),).cuda()
        # change the above line to the following line to use the CPU
        self.PGE_module=nn.Sequential(nn.Linear(self.x.shape[-1], num_nodes),)
        self.mask = self.construct_edge_mask(num_nodes)
        # create an empty self.masked_adj
        self.masked_adj = adj

        params = [self.mask]
        if args.explainer_backbone!='GNNExplainer':
            params.extend(self.PGE_module.parameters()) # an nn.Linear module has two parameters: the weight matrix and the bias vector. So there are 3 elements in the list params after this line of code

        self.optimizer = util.build_optimizer(args, params)

    def construct_edge_mask(self, num_nodes):
        mask = nn.Parameter(torch.FloatTensor(num_nodes, num_nodes))
        std = nn.init.calculate_gain("relu") * math.sqrt(2.0 / (num_nodes + num_nodes)) # This line of code initializes the standard deviation (std) of the weight tensor for a neural network layer using a technique called Xavier initialization, which aims to set the initial weights of the network in a way that helps to avoid the vanishing and exploding gradient problems during training.
        with torch.no_grad():
            mask.normal_(1.0, std)
            # mask.data += self.adj * 10
        return mask

    def _masked_adj(self): # the mask should be symmetric
        if self.args.explainer_backbone=='PGExplainer':
            sym_mask = self.PGE_module(self.x).squeeze(0) # (1, num_nodes, num_nodes) -> (num_nodes, num_nodes)
        else:
            sym_mask = self.mask

        sym_mask = torch.sigmoid(sym_mask) # apply the sigmoid function to the mask, mapping every element of the mask to a value between 0 and 1
        sym_mask = (sym_mask + sym_mask.t()) / 2 # now sym_mask is a symmetric matrix
        adj = self.adj.cuda() if self.args.gpu else self.adj

        self.sym_mask=sym_mask
        masked_adj = adj * sym_mask
        return masked_adj

    def forward(self, node_idx):
        x = self.x.cuda() if self.args.gpu else self.x # feature matrix of nodes in the computational graph of the node to be explained
        # self.masked_adj is calculated using self.mask, which is the param to be optimized, when backbone='GNNExplainer'
        self.masked_adj = self._masked_adj() # the adjacency matrix of the computational graph of the node to be explained
        ypred, adj_att = self.model(x, self.masked_adj) # adj_att == 0
        node_pred = ypred[0,node_idx, :] # the prediction of the node to be explained
        res = nn.Softmax(dim=0)(node_pred) # this is what the paper calls maintainging sum of y𝑖[𝑘] = 1 in page 318
        return res, adj_att

    def loss(self):
        x = self.x.cuda() if self.args.gpu else self.x
        self.masked_adj = self._masked_adj()
        new_adj = self.masked_adj # not 0-1 matrix
        self.model.eval()
        # tensor_pred = self.model(x, new_adj)[0].unsqueeze(0)
        tensor_pred = self.model(x, new_adj)[0].flatten(start_dim=0, end_dim=-2)
        wass_dis = 0
        sens = np.array(self.sens)
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
                # pred1 = tensor_pred[0][index1]
                # pred2 = tensor_pred[0][index2]
                pred1 = tensor_pred[index1]
                pred2 = tensor_pred[index2]
                # compute the Wasserstein distance between the predictions of the nodes with the current pair of values
                wass_dis = wass_dis + wasserstein(pred1, pred2)[0]

        return -wass_dis


