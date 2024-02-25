from .approximator import grad_z_graph, cal_influence_graph, s_test_graph_cost, cal_influence_graph_nodal
import numpy as np
import scipy.sparse as sp
import torch
import dgl
from tqdm import tqdm
import pandas as pd
from scipy.spatial import distance_matrix
import os
import networkx as nx
import time
# import argparse
# from torch_geometric.utils import convert
import warnings
warnings.filterwarnings('ignore')
import ctypes
# ctypes.cdll.LoadLibrary('caffe2_nvrtc.dll')
torch.backends.cudnn.benchmark = True


def find123Nei(G, node):
    nodes = list(nx.nodes(G))
    nei1_li = []
    nei2_li = []
    nei3_li = []
    for FNs in list(nx.neighbors(G, node)):
        nei1_li .append(FNs)

    for n1 in nei1_li:
        for SNs in list(nx.neighbors(G, n1)):
            nei2_li.append(SNs)
    nei2_li = list(set(nei2_li) - set(nei1_li))
    if node in nei2_li:
        nei2_li.remove(node)

    for n2 in nei2_li:
        for TNs in nx.neighbors(G, n2):
            nei3_li.append(TNs)
    nei3_li = list(set(nei3_li) - set(nei2_li) - set(nei1_li))
    if node in nei3_li:
        nei3_li.remove(node)

    return nei1_li, nei2_li, nei3_li


def feature_norm(features):
    min_values = features.min(axis=0)[0]
    max_values = features.max(axis=0)[0]
    return 2*(features - min_values).div(max_values-min_values) - 1


def build_relationship(x, thresh=0.25):
    df_euclid = pd.DataFrame(1 / (1 + distance_matrix(x.T.T, x.T.T)), columns=x.T.columns, index=x.T.columns)
    df_euclid = df_euclid.to_numpy()
    idx_map = []
    for ind in range(df_euclid.shape[0]):
        max_sim = np.sort(df_euclid[ind, :])[-2]
        neig_id = np.where(df_euclid[ind, :] > thresh * max_sim)[0]
        import random
        random.seed(912)
        random.shuffle(neig_id)
        for neig in neig_id:
            if neig != ind:
                idx_map.append([ind, neig])
    idx_map = np.array(idx_map)

    return idx_map


def get_adj(dataset_name):
    predict_attr = "RECID"
    if dataset_name == 'bail':
        predict_attr="RECID"
    elif dataset_name == 'income':
        predict_attr = "income"

    if dataset_name == 'pokec1' or dataset_name == 'pokec2':
        if dataset_name == 'pokec1':
            edges = np.load('../data/pokec_dataset/region_job_1_edges.npy')
            labels = np.load('../data/pokec_dataset/region_job_1_labels.npy')
        else:
            edges = np.load('../data/pokec_dataset/region_job_2_2_edges.npy')
            labels = np.load('../data/pokec_dataset/region_job_2_2_labels.npy')

        adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                            shape=(labels.shape[0], labels.shape[0]),
                            dtype=np.float32)
        adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
        return adj

    path="../data/" + str(dataset_name) + "/"
    dataset = dataset_name
    print('Reconstructing the adj of {} dataset...'.format(dataset))

    idx_features_labels = pd.read_csv(os.path.join(path, "{}.csv".format(dataset)))
    header = list(idx_features_labels.columns)
    header.remove(predict_attr)

    if os.path.exists(f'{path}/{dataset}_edges.txt'):
        edges_unordered = np.genfromtxt(f'{path}/{dataset}_edges.txt').astype('int')
    else:
        edges_unordered = build_relationship(idx_features_labels[header], thresh=0.6)
        np.savetxt(f'{path}/{dataset}_edges.txt', edges_unordered)

    features = sp.csr_matrix(idx_features_labels[header], dtype=np.float32)
    labels = idx_features_labels[predict_attr].values

    idx = np.arange(features.shape[0])
    idx_map = {j: i for i, j in enumerate(idx)}
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=int).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    return adj


def del_adj(harmful, adj_vanilla):
    adj = adj_vanilla
    mask = np.ones(adj.shape[0], dtype=bool)
    mask[harmful] = False
    # Get the indices and values of the non-zero elements
    indices = adj.coalesce().indices().cpu().numpy()
    values = adj.coalesce().values().cpu().numpy()

    # Convert the indices to a 2D array where each row is the indices of a non-zero element
    indices = indices.T

    # Create the sparse COO matrix
    adj_coo = sp.coo_matrix((values, (indices[: , 0], indices[: , 1])), shape=adj.shape)
    adj = sp.coo_matrix(adj_coo.tocsr()[mask,:][:,mask])

    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
    adj = adj + sp.eye(adj.shape[0])

    return adj


def pdd(adj, idx_train, features, idx_test, labels, sens, idx_val, model, g):
    adj_vanilla = adj.clone()
    # edge_index = adj.coalesce().indices()
    print("Pre-processing data...")
    computation_graph_involving = []
    eye_sparse = torch.eye(adj.shape[0]).to_sparse()
    the_adj = adj - eye_sparse
    edge_list = the_adj.coalesce().indices()
    # convert edge_list to np.array
    edge_list = edge_list.numpy()
    # convert edge_list to a list of 2-tuples
    edge_list = list(zip(edge_list[0, :], edge_list[1, :]))

    hop = 1
    G = nx.Graph(edge_list)
    idx_train_vanilla = idx_train.clone()
    for i in tqdm(range(idx_train_vanilla.shape[0])):
        neighbors = find123Nei(G, idx_train_vanilla[i].item())
        mid = []
        for j in range(hop):
            mid += neighbors[j]
        mid = list(set(mid).intersection(set(idx_train_vanilla.numpy().tolist())))
        computation_graph_involving.append(mid)
    print("Pre-processing completed.")

    time1 = time.time()
    features_vanilla = features
    idx_test_vanilla = idx_test.clone()
    labels_vanilla = labels
    sens_vanilla = sens
    idx_val_vanilla = idx_val.clone()
    h_estimate_cost = s_test_graph_cost(g, features_vanilla, idx_train_vanilla, 
                                        idx_test_vanilla, labels_vanilla, sens_vanilla, model, gpu=0)
    gradients_list = grad_z_graph(g, features_vanilla, idx_train_vanilla, labels_vanilla, 
                                model, gpu=0)
    influence, harmful, helpful, harmful_idx, helpful_idx = cal_influence_graph(idx_train_vanilla, h_estimate_cost, gradients_list, gpu=0)
    non_iid_influence = []

    for i in tqdm(range(idx_train_vanilla.shape[0])):

        if len(computation_graph_involving[i]) == 0:
            non_iid_influence.append(0)
            continue

        reference = list(range(adj_vanilla.shape[0]))
        for j in range(len(reference) - idx_train_vanilla[i]):
            reference[j + idx_train_vanilla[i]] -= 1

        mask = np.ones(idx_train_vanilla.shape[0], dtype=bool)
        mask[i] = False
        idx_train = idx_train_vanilla[mask]
        idx_val = idx_val_vanilla.clone()
        idx_test = idx_test_vanilla.clone()
        # print(idx_train)
        idx_train = torch.LongTensor(np.array(reference)[idx_train.numpy()])
        # print(idx_train)
        idx_val = torch.LongTensor(np.array(reference)[idx_val.numpy()])
        idx_test = torch.LongTensor(np.array(reference)[idx_test.numpy()])

        computation_graph_involving_copy = computation_graph_involving.copy()
        for j in range(len(computation_graph_involving_copy)):
            computation_graph_involving_copy[j] = np.array(reference)[computation_graph_involving_copy[j]]

        mask = np.ones(labels_vanilla.shape[0], dtype=bool)
        mask[idx_train_vanilla[i]] = False

        features = features_vanilla[mask, :]
        labels = labels_vanilla[mask]
        sens = sens_vanilla[mask]

        adj = del_adj(idx_train_vanilla[i], adj_vanilla)
        adj_coo = adj.tocoo()
        src, dst = adj_coo.row, adj_coo.col

        # Create the heterograph
        g = dgl.heterograph({('node', 'edge', 'node'): (src, dst)})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        g = g.int().to(device)
        # edge_index = convert.from_scipy_sparse_matrix(adj)[0]
        h_estimate_cost_nodal = h_estimate_cost.copy()
        gradients_list_nodal = grad_z_graph(g, features, torch.LongTensor(computation_graph_involving_copy[i]), labels, model, gpu=0)
        influence_nodal, _, _, _, _ = cal_influence_graph_nodal(idx_train, torch.LongTensor(computation_graph_involving_copy[i]), h_estimate_cost_nodal, gradients_list_nodal,
                                                                                    gpu=0)
        non_iid_influence.append(sum(influence_nodal))

    final_influence = []
    for i in range(len(non_iid_influence)):
        ref = [idx_train_vanilla.numpy().tolist().index(item) for item in (computation_graph_involving[i] + [idx_train_vanilla[i]])]
        final_influence.append(non_iid_influence[i] - np.array(influence)[ref].sum())

    time4 = time.time()
    print("Average time per training node:", (time4 - time1)/1000, "s")
    # np.save('final_influence.npy', np.array(final_influence))
    # NOTICE: here the helpfulness means the helpfulness to UNFAIRNESS
    # print("Helpfulness : ")
    # print(final_influence)
    # print(np.argsort(final_influence))

    return np.array(final_influence)