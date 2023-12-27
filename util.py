import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import pickle
import networkx as nx
from networkx.readwrite import json_graph
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import sys
import base64
import importlib.util
import inspect

from emug.server.explainers import explain_unfairness_global
from emug.server.explainers import explain_unfairness_local
from emug.server.explainers import explain_unfairness
from emug.server.explainers import configs
from emug.server.explainers import util

torch.manual_seed(42)

# configuration
PERPLEXITY = 15 # german
# PERPLEXITY = 150 # artificial

class EmbeddingHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.embedding = output[0].detach().clone()

    def remove(self):
        self.hook.remove()


def all_embeddings(model, x, adj):
    # hooks = [util.EmbeddingHook(layer) for layer in model.children()]
    hooks = []
    for layer in model.children():
        # check if layer is a nn.Module.loss
        if 'loss' not in str(type(layer)): # check if layer is a nn.Module.loss
            if isinstance(layer, nn.ModuleList):
                for sublayer in layer:
                    hooks.append(EmbeddingHook(sublayer))
            else:
                hooks.append(EmbeddingHook(layer))
    # Forward pass
    output = model(x, adj)
    # Access embeddings of each layer
    layer_embeddings = [hook.embedding for hook in hooks]
    for i, emb in enumerate(layer_embeddings):
        if emb.shape[0] == 1:
            # get rid of the batch dimension
            layer_embeddings[i] = emb.squeeze(0)
    # Remove hooks
    for hook in hooks:
        hook.remove()
    return layer_embeddings


def proj_emb(embeddings_ls, projector): 
    '''
    Project embeddings to 2D space
    Args:
        embeddings_ls: list of embeddings
        projector: sklearn.manifold.TSNE or sklearn.decomposition.PCA
    Returns:
        embeddings_2d_ls: list of 2D embeddings
    '''  
    embeddings_2d_ls = []
    # begin projection
    print('\nbegin projection...')
    for e in tqdm(embeddings_ls):
        # if e.shape[1] != 2:
        if True:
            embeddings_2d = projector.fit_transform(e)
            embeddings_2d_ls.append(embeddings_2d)
        else:
            embeddings_2d_ls.append(e)
    # finish projection
    print('\nfinish projection...')
    return embeddings_2d_ls


def proj_emb_based_on_earlier_proj(embeddings_ls, earlier_embeddings_2d_ls, projection_type='tsne', perplexity=PERPLEXITY, 
                                   n_components=2, n_iter=1250, learning_rate='auto'):
    embeddings_2d_ls = []
    # begin projection
    print('\nbegin projection...')
    # for e in tqdm(embeddings_ls):
    for i, e in tqdm(enumerate(embeddings_ls)):
        if e.shape[1] != 2:
            if projection_type == 'tsne':
                projector = TSNE(perplexity=perplexity, n_components=n_components, n_iter=n_iter, 
                                 learning_rate=learning_rate, init=earlier_embeddings_2d_ls[i], random_state=42)
            embeddings_2d = projector.fit_transform(e)
            embeddings_2d_ls.append(embeddings_2d)
        else:
            embeddings_2d_ls.append(e)
    # finish projection
    print('\nfinish projection...')
    return embeddings_2d_ls


# def get_graph_dict():
#     # Create a NetworkX graph from the adjacency matrix
#     G = nx.from_numpy_matrix(np.triu(np.array(data['adj']), k=1))
#     # Compute the force-directed layout using the Fruchterman-Reingold algorithm
#     pos = nx.spring_layout(G, scale=1, iterations=70, seed=2)
#     # pos = nx.spring_layout(G, scale=1, iterations=200, seed=2) # artificial
#     # Create a dictionary for the node positions
#     nodes = [{'id': str(node), 'x': pos[node][0], 'y': pos[node][1]} for node in G.nodes()]
#     # Create a dictionary for the edges
#     edges = [{'source': str(edge[0]), 'target': str(edge[1])} for edge in G.edges()]
#     # Create the final dictionary or JSON object
#     graph_dict = {'nodes': nodes, 'edges': edges, 'scaled': False}
#     return graph_dict


def compute_layout_by_edges(sources, targets, view_width, view_height):
    # Define the graph
    G = nx.Graph()

    # Add edges from the source and target arrays
    edges = np.vstack((sources, targets)).T
    G.add_edges_from(edges)

    # Generate the layout of the graph using spring_layout
    pos = nx.spring_layout(G, k=0.15, iterations=50, seed=42)
    # Create a dictionary for the node positions
    nodes = [{'id': str(node), 
              'x': pos[node][0]*view_width/2.0 + view_width/2.0, 
              'y': pos[node][1]*view_height/2.0 + view_height/2.0} for node in G.nodes()]
    # Create a dictionary for the edges
    edges = [{'source': str(edge[0]), 'target': str(edge[1])} for edge in G.edges()]
    # Create the final dictionary or JSON object
    graph_dict = {'nodes': nodes, 'edges': edges}

    return graph_dict


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
    # M_drop = F.dropout(M, 0.0 / (nx * ny))
    M_drop = F.dropout(M, 0)
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


def model_based_fairness(embeddings, sens):
    # sens is a list of strs, create a list representing the unique sensitive attributes, and map each sensitive attribute to a number
    sens_unique = list(set(sens))
    mapping = {value: index for index, value in enumerate(sens_unique)}
    sens_int = [mapping[value] for value in sens]

    # split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(embeddings, sens_int, test_size=0.2, random_state=42)
    # train the MLP classifier
    clf = MLPClassifier(hidden_layer_sizes=(100, 100), max_iter=1000, alpha=1e-4,
                        solver='sgd', verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1) # (100, 100) means the MLPClassifier will have two hidden layers, each with 100 neurons
    # don't print the training process
    clf.verbose = False
    clf.fit(X_train, y_train)
    # predict the sensitive attributes
    y_pred = clf.predict(X_test)
    n_class = len(np.unique(sens_unique))
    return (accuracy_score(y_test, y_pred), 
            recall_score(y_test, y_pred, average='macro'),
            precision_score(y_test, y_pred, average='weighted'),
            f1_score(y_test, y_pred, average='weighted'),
            roc_auc_score(y_test, clf.predict_proba(X_test) if n_class > 2 else y_pred, multi_class='ovo'))


def subgroup_unfair_metrics(sens_names, sens, e):
    ret = pd.DataFrame()
    # calculate wasserstein distance between embeddings of different sensitive groups
    for sens_name in sens_names:
        # metric[sens_name] = {}
        # metric[sens_name]['wasserstein'] = {}
        sens_idx = sens_names.index(sens_name)
        s = np.array(sens[sens_idx])
        # get unique value in s
        s_unique = np.unique(s)
        acc, recall, precision, f1, auc = model_based_fairness(e, s)
        ret[sens_name + '-model_based-' + 'acc'] = [acc]
        ret[sens_name + '-model_based-' + 'recall'] = [recall]
        ret[sens_name + '-model_based-' + 'precision'] = [precision]
        ret[sens_name + '-model_based-' + 'f1'] = [f1]
        ret[sens_name + '-model_based-' + 'auc'] = [auc]
        # metric[sens_name]['model_based'] = {'acc': acc,
        #                                     'recall': recall,
        #                                     'precision': precision,
        #                                     'f1': f1,
        #                                     'auc': auc,
        #                                     }
        
        # loop through each pair of unique values
        for i in range(len(s_unique)):
            for j in range(i+1, len(s_unique)):
                # get indices of s that are equal to s_unique[i] and s_unique[j]
                idx_i = np.where(s == s_unique[i])[0]
                idx_j = np.where(s == s_unique[j])[0]
                # calculate wasserstein distance
                w_dist = float(wasserstein(e[idx_i], e[idx_j])[0])

                # metric[sens_name]['wasserstein'][str(int(s_unique[i])) + '_' + str(int(s_unique[j]))] = w_dist
                ret[sens_name + '-wasserstein-' + str(int(s_unique[i])) + '_' + str(int(s_unique[j]))] = [w_dist]

    return ret


def neighborhoods_each(adj, max_hop):
    neighborhoods = []
    neighborhoods.append(adj)
    for i in range(max_hop):
        neighborhoods.append(neighborhoods[-1].matmul(adj))
    return neighborhoods


def init(model, adj, feat, sens, sens_names):
    data = {}
    adj = torch.tensor(adj, dtype=torch.float)
    feat = torch.tensor(feat, requires_grad=False, dtype=torch.float)

    embeddings = all_embeddings(model, feat, adj)
    adj_diag0 = adj.clone()
    adj_diag0 = adj_diag0 - np.diag(np.diag(adj_diag0))
    embeddings_diag0 = all_embeddings(model, feat, adj_diag0)

    adj_iid = torch.eye(adj.shape[0])
    embeddings_iid = all_embeddings(model, feat, adj_iid)

    # check whether the type of sens is list
    if type(sens) is not list:
        sens = sens.tolist()
    # check whether the type of sens_names is list
    if type(sens_names) is not list:
        sens_names = sens_names.tolist()

    data['model'] = model
    data['adj'] = adj
    data['feat'] = feat
    data['sens'] = sens
    data['sens_names'] = sens_names
    data['embeddings'] = embeddings
    data['embeddings_diag0'] = embeddings_diag0
    data['embeddings_iid'] = embeddings_iid

    # graph_dict = get_graph_dict()
    degrees = np.array(np.sum(data['adj'].numpy(), axis=1)).reshape(-1)
    degree_boxes = pd.qcut(degrees, q=3, labels=[0, 1, 2]).tolist()
    # neighborhoods_123 = neighborhoods_each(data['adj'].clone().detach(), 3)

    return {'embeddings': embeddings,
            'embeddings_diag0': embeddings_diag0,
            'embeddings_iid': embeddings_iid,
            # 'graph_dict': graph_dict,
            'degrees': degrees,
            'degree_boxes': degree_boxes,}


def calc_metrics(embeddings, sens_names, sens, degree_boxes):
    metrics = []
    for e in embeddings:
        # metric = {}
        # metric['sens'] = {}
        # subgroup_unfair_metrics(metric['sens'], sens_names, sens, e)
        # subgroup_unfair_metrics(metric, ['degree'], [degree_boxes], e)                  
        # metrics.append(metric)
        s = subgroup_unfair_metrics(sens_names, sens, e)
        d = subgroup_unfair_metrics(['degree'], [degree_boxes], e) 
        metric = pd.concat([s, d], axis=1)
        metrics.append(metric)

    return metrics

    # model = data['model']
    # feat = data['feat']
    # adj_diag0 = data['adj'].clone()
    # adj_diag0 = adj_diag0 - np.diag(np.diag(adj_diag0))
    # embeddings_diag0 = all_embeddings(model, feat, adj_diag0)
    # metrics_diag0 = []
    # for e in embeddings_diag0:
    #     metric = {}
    #     metric['sens'] = {}
    #     subgroup_unfair_metrics(metric['sens'], sens_names, sens, e)
    #     subgroup_unfair_metrics(metric, ['degree'], [degree_boxes], e)                  
    #     metrics_diag0.append(metric)

    # # create an identity matrix with shape of (n, n), and store it in adj_iid
    # adj_iid = torch.eye(data['adj'].shape[0])
    # # get embeddings of adj_iid
    # embeddings_iid = all_embeddings(model, feat, adj_iid)
    # metrics_iid = []
    # for e in embeddings_iid:
    #     metric = {}
    #     metric['sens'] = {}
    #     subgroup_unfair_metrics(metric['sens'], sens_names, sens, e)
    #     subgroup_unfair_metrics(metric, ['degree'], [degree_boxes], e)                  
    #     metrics_iid.append(metric)

    # return {'metrics': metrics,
    #         'metrics_diag0': metrics_diag0,
    #         'metrics_iid': metrics_iid,
    #         'sens_names': data['sens_names'],}


def explanation_level_local_watch_debias_view(adj, max_hop):
    neighborhoods_each(adj, max_hop)

    ids = list(map(int, request.json['nodeIds']))
    threshold = float(request.json['threshold'])
    hops = int(request.json['hops'])
    height_subgraph = request.json['heightSubgraph'] * 0.99
    width_subgraph = request.json['widthSubgraph'] * 0.99
    sens_name_idx = int(request.json['sensNameIdx'])

    prog_args = configs.arg_parse_explain()
    prog_args.dataset = 'german'
    prog_args.explainer_backbone = 'GNNExplainer'
    prog_args.method = 'gat'
    prog_args.num_gc_layers = hops

    sens = data['sens']
    adj = torch.tensor(data['adj']).float()
    neighborhoods = neighborhoods_123[hops - 1]
    neighborhoods = torch.where(neighborhoods != 0, torch.tensor(1), torch.tensor(0))

    # convert np.array ckpt['feat'] to torch.tensor
    feat = torch.tensor(data['feat']).float()
    sens_names = data['sens_names']

    model = data['model']
    # initialize explainer
    explainer = explain_unfairness_local.LocalExplainer(model=model, adj=adj, feat=feat, args=prog_args, neighborhoods=neighborhoods)

    masked_adj_local, neighbors = explainer.explain(sens[sens_name_idx], ids) # where the algorithm is implemented
    
    indices = np.triu(masked_adj_local, k=1) > threshold
    sources, targets = np.where(indices)
    scores = masked_adj_local[indices].tolist()
    # sources, targets = sources.tolist(), targets.tolist()
    n_edges = len(sources)
    mask = [{'source': str(int(neighbors[sources[i]])), 
             'target': str(int(neighbors[targets[i]])), 
             'score': scores[i]} for i in range(n_edges)]

    # create a list of lists named hop_indicator in which each list contains the i-th hop neighbors of node whose index is id, the higher hop neighbors do not include the lower hop neighbors
    hop_indicator = []
    hop_indicator.append(ids)
    for i in range(hops):
        current_neighborhoods = neighborhoods_123[i]
        neighbors_adj_row = current_neighborhoods[ids, :].sum(dim=0)
        tmp_indicater = neighbors_adj_row.nonzero().squeeze().tolist()
        # take the difference set of all the formers
        for j in range(i + 1):
            tmp_indicater = list(set(tmp_indicater) - set(hop_indicator[j]))
        hop_indicator.append(tmp_indicater)
    # pop the first element
    # hop_indicator.pop(0)
    
    # map lambda x: str(int(neighbors[x])) to sources and convert to np.array
    sources_total = np.array(list(map(lambda x: str(int(neighbors[x])), sources)))
    targets_total = np.array(list(map(lambda x: str(int(neighbors[x])), targets)))
    subgraph = compute_layout_by_edges(sources_total, targets_total, width_subgraph*0.95, height_subgraph*0.95)

    ret = {'mask': mask, 
           'neighbors': list(map(str, neighbors)), 
           'hop_indicator': hop_indicator,
           'subgraph': subgraph,}
    return ret