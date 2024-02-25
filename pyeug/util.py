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


torch.manual_seed(42)

# configuration
PERPLEXITY = 15 # german
# PERPLEXITY = 150 # artificial

class EmbeddingHook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # self.embedding = output[0].detach().clone()
        self.embedding = output[0]

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
    # print('\nbegin projection...')
    # for e in tqdm(embeddings_ls):
    for e in embeddings_ls:
        # if e.shape[1] != 2:
        if isinstance(e, torch.Tensor):
            e = e.detach().cpu().numpy()
            embeddings_2d = projector.fit_transform(e)
            embeddings_2d_ls.append(embeddings_2d)
    # finish projection
    # print('\nfinish projection...')
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
    adj_torch = torch.tensor(adj, dtype=torch.float)
    neighborhoods = []
    neighborhoods.append(adj_torch)
    for i in range(max_hop):
        neighborhoods.append(neighborhoods[-1].matmul(adj_torch))
        # neighborhoods.append(neighborhoods[-1] @ adj)
    return neighborhoods


# def init(model, adj, feat, sens, sens_names, g, emb_fnc):
def init(model, feat, g, emb_fnc):
    # data = {}
    # adj = torch.tensor(adj, dtype=torch.float)
    # feat = torch.tensor(feat, requires_grad=False, dtype=torch.float)

    # embeddings = all_embeddings(model, feat, adj)
    embeddings = emb_fnc(g, feat, model)
    embeddings = [e.detach().clone() for e in embeddings]
    # adj_diag0 = adj.clone()
    # adj_diag0 = adj_diag0 - np.diag(np.diag(adj_diag0))
    # embeddings_diag0 = all_embeddings(model, feat, adj_diag0)
    # embeddings_diag0 = [e.detach().clone() for e in embeddings_diag0]

    # adj_iid = torch.eye(adj.shape[0])
    # embeddings_iid = all_embeddings(model, feat, adj_iid)
    # embeddings_iid = [e.detach().clone() for e in embeddings_iid]

    # # check whether the type of sens is list
    # if type(sens) is not list:
    #     sens = sens.tolist()
    # # check whether the type of sens_names is list
    # if type(sens_names) is not list:
    #     sens_names = sens_names.tolist()

    # data['model'] = model
    # data['adj'] = adj
    # data['feat'] = feat
    # data['sens'] = sens
    # data['sens_names'] = sens_names
    # data['embeddings'] = embeddings
    # data['embeddings_diag0'] = embeddings_diag0
    # data['embeddings_iid'] = embeddings_iid

    # graph_dict = get_graph_dict()
    # degrees = np.array(np.sum(data['adj'].numpy(), axis=1)).reshape(-1)
    # degree_boxes = pd.qcut(degrees, q=3, labels=[0, 1, 2]).tolist()
    # neighborhoods_123 = neighborhoods_each(data['adj'].clone().detach(), 3)

    return {'embeddings': embeddings,
            # 'embeddings_diag0': embeddings_diag0,
            # 'embeddings_iid': embeddings_iid,
            # 'graph_dict': graph_dict,
            # 'degrees': degrees,
            # 'degree_boxes': degree_boxes,
            }


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


def force_directed_layout_per_layer(G, layers, current_part, n_y_part, iterations=50, extra_layer=2, extra_layer_y=1):
    # print('number of nodes in G in force_directed_layout_per_layer: ', len(G.nodes()))
    # print('G.nodes: ', G.nodes())
    pos = {}
    # create a n by 2 np array, where n is the number of nodes in G
    # pos = np.zeros((len(G.nodes), 2))
    n_layers = len(layers)
    # n_rows_cols = math.ceil(math.sqrt(n_layers))
    for i, layer in enumerate(layers):
        # print('nodes in layer: ', layer)
        layer_subgraph = G.subgraph(layer)
        # print('nodes of layer_subgraph: ', layer_subgraph.nodes())
        layer_pos = nx.spring_layout(layer_subgraph, iterations=iterations)
        # print('layer_pos: ', layer_pos)
        x_shift = -1 + 2 / n_layers * (i + 0.5)
        # y_shift = 0
        y_shift = -1 + 2 / n_y_part * (current_part + 0.5)
        scaled_layer_pos = {k: [x + y for x, y in zip([v[0] / (n_layers + extra_layer), v[1] / (n_y_part + extra_layer_y)], [x_shift, y_shift])] for k, v in layer_pos.items()}
        pos.update(scaled_layer_pos)
        # for k, v in layer_pos.items():
            # find the index of k
    return pos


# def force_directed_layout_per_layer_old(G, layers, iterations=50, extra_layer=2):
#     # print('number of nodes in G in force_directed_layout_per_layer: ', len(G.nodes()))
#     # print('G.nodes: ', G.nodes())
#     pos = {}
#     # create a n by 2 np array, where n is the number of nodes in G
#     # pos = np.zeros((len(G.nodes), 2))
#     n_layers = len(layers)
#     # n_rows_cols = math.ceil(math.sqrt(n_layers))
#     for i, layer in enumerate(layers):
#         # print('nodes in layer: ', layer)
#         layer_subgraph = G.subgraph(layer)
#         # print('nodes of layer_subgraph: ', layer_subgraph.nodes())
#         layer_pos = nx.spring_layout(layer_subgraph, iterations=iterations)
#         # print('layer_pos: ', layer_pos)
#         x_shift = -1 + 2 / n_layers * (i + 0.5)
#         y_shift = 0
#         scaled_layer_pos = {k: [x + y for x, y in zip([v[0] / (n_layers + extra_layer), v[1]], [x_shift, y_shift])] for k, v in layer_pos.items()}
#         pos.update(scaled_layer_pos)
#         # for k, v in layer_pos.items():
#             # find the index of k
#     return pos


# def hopwise_force_directed_layout_old(explain_induced_node_indices, explain_induced_node_hops, source_explained, target_explained,
#     extra_layer=2):
#     # create a graph
#     G = nx.Graph()
#     # add nodes
#     G.add_nodes_from(explain_induced_node_indices)
#     # add edges
#     edges = np.stack((source_explained, target_explained), axis=1)
#     G.add_edges_from(edges)
#     # get the unique values of explain_induced_node_hops
#     unique_hops = np.unique(explain_induced_node_hops)
#     # create a list of lists, where each list contains the indices of nodes with the same hop
#     layers = [explain_induced_node_indices[np.where(explain_induced_node_hops == hop)[0]] for hop in unique_hops]
#     # if len(layers) > 0:
#     #     print('len of explain_induced_node_indices: ', len(explain_induced_node_indices))
#     #     print('len of unique values in explain_induced_node_indices: ', len(np.unique(explain_induced_node_indices)))
#     #     print('len of flattened layers: ', sum([len(layer) for layer in layers]))
#     #     print('len of unique values in layers: ', len(np.unique(np.concatenate(layers))))
#     #     print('number of nodes in G: ', len(G.nodes()))
#     # print('layers in hopwise_force_directed_layout: ', layers)
#     pos = force_directed_layout_per_layer(G, layers, extra_layer)
#     # create a n by 2 np array, where n is the number of rows in explain_induced_node_indices
#     ret = np.zeros((explain_induced_node_indices.shape[0], 2))
#     # print(ret.shape)
#     # find the index of each key of pos in explain_induced_node_indices
#     # print('the number of elements in pos: ', len(pos))
#     # print('len of explain_induced_node_indices: ', len(explain_induced_node_indices))
#     # print('pos: ', pos)
#     # print('explain_induced_node_indices: ', explain_induced_node_indices)
#     for k, v in pos.items():
#         i = np.where(explain_induced_node_indices == k)[0][0]
#         ret[i] = v

#     unique_hops_max = unique_hops.max() if len(unique_hops) else -1
#     if len(ret):
#         return ret[: , 0], ret[: , 1], unique_hops_max
#     else:
#         return np.array([]), np.array([]), unique_hops_max


def hopwise_force_directed_layout(explain_induced_node_indices, explain_induced_node_hops, 
    explain_induced_node_indices_unfair_pure, explain_induced_node_indices_fair_pure, explain_induced_node_indices_intersection, 
    source_explained, target_explained,
    extra_layer=2):
    print('the node ids in 0 hop: ', explain_induced_node_indices[explain_induced_node_hops == 0])

    # Create masks for both columns using logical indexing
    mask_source_unfair = np.isin(source_explained, explain_induced_node_indices_unfair_pure)
    mask_target_unfair = np.isin(target_explained, explain_induced_node_indices_unfair_pure)
    mask_source_fair = np.isin(source_explained, explain_induced_node_indices_fair_pure)
    mask_target_fair = np.isin(target_explained, explain_induced_node_indices_fair_pure)
    mask_source_intersection = np.isin(source_explained, explain_induced_node_indices_intersection)
    mask_target_intersection = np.isin(target_explained, explain_induced_node_indices_intersection)

    # Combine the masks using logical AND
    mask_unfair = mask_source_unfair & mask_target_unfair
    mask_fair = mask_source_fair & mask_target_fair
    mask_intersection = mask_source_intersection & mask_target_intersection

    # Use the combined mask to select the rows from source and target
    source_explained_unfair = source_explained[mask_unfair]
    target_explained_unfair = target_explained[mask_unfair]
    source_explained_fair = source_explained[mask_fair]
    target_explained_fair = target_explained[mask_fair]
    source_explained_intersection = source_explained[mask_intersection]
    target_explained_intersection = target_explained[mask_intersection]

    explain_induced_node_hops_unfair = explain_induced_node_hops[np.isin(explain_induced_node_indices, explain_induced_node_indices_unfair_pure)]
    explain_induced_node_hops_fair = explain_induced_node_hops[np.isin(explain_induced_node_indices, explain_induced_node_indices_fair_pure)]
    explain_induced_node_hops_intersection = explain_induced_node_hops[np.isin(explain_induced_node_indices, explain_induced_node_indices_intersection)]

    unique_hops = np.unique(explain_induced_node_hops)

    ret_unfair = hopwise_force_directed_layout_single(explain_induced_node_indices_unfair_pure, explain_induced_node_hops_unfair,
        source_explained_unfair, target_explained_unfair, 0, 
        unique_hops, explain_induced_node_indices,
        extra_layer=extra_layer)
    ret_intersection = hopwise_force_directed_layout_single(explain_induced_node_indices_intersection, explain_induced_node_hops_intersection,
        source_explained_intersection, target_explained_intersection, 1,
        unique_hops, explain_induced_node_indices,
        extra_layer=extra_layer)
    ret_fair = hopwise_force_directed_layout_single(explain_induced_node_indices_fair_pure, explain_induced_node_hops_fair,
        source_explained_fair, target_explained_fair, 2,
        unique_hops, explain_induced_node_indices,
        extra_layer=extra_layer)

    ret = ret_unfair + ret_intersection + ret_fair

    unique_hops_max = unique_hops.max() if len(unique_hops) else -1
    if len(ret):
        return ret[: , 0], ret[: , 1], unique_hops_max
    else:
        return np.array([]), np.array([]), unique_hops_max


def hopwise_force_directed_layout_single(explain_induced_node_indices_part, explain_induced_node_hops_part, 
    source_explained, target_explained, current_part,
    unique_hops, explain_induced_node_indices,
    extra_layer=2, n_y_part=3):
    # create a graph
    G = nx.Graph()
    # add nodes
    G.add_nodes_from(explain_induced_node_indices_part)
    # add edges
    edges = np.stack((source_explained, target_explained), axis=1)
    G.add_edges_from(edges)

    # create a list of lists, where each list contains the indices of nodes with the same hop
    layers = [explain_induced_node_indices_part[np.where(explain_induced_node_hops_part == hop)[0]] for hop in unique_hops]

    pos = force_directed_layout_per_layer(G, layers, current_part, n_y_part, extra_layer)
    # create a n by 2 np array, where n is the number of rows in explain_induced_node_indices
    ret = np.zeros((explain_induced_node_indices.shape[0], 2))

    for k, v in pos.items():
        i = np.where(explain_induced_node_indices == k)[0][0]
        ret[i] = v

    return ret


def concatenate_elements(*args):
    return "-".join(map(str, args))


def check_unique_values(arr):
    # Initialize an array of False values with the same number of columns as arr
    unique_check = np.full(arr.shape[1], False, dtype=bool)

    # Iterate through each column
    for i in range(arr.shape[1]):
        # Check if the number of unique values in the column is less than or equal to 2
        if len(np.unique(arr[:, i])) <= 2:
            unique_check[i] = True

    return unique_check