import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import torch
import dgl
from tqdm import tqdm
import time
from torch.profiler import profile, record_function, ProfilerActivity

def individual_bias_contribution(adj, features, model, group):
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_gpus = torch.cuda.device_count()  # Get the number of GPUs available
    current_device = 0
    torch.cuda.set_device(current_device)  # Set initial GPU
    device = torch.device(f'cuda:{current_device}' if torch.cuda.is_available() else 'cpu')

    # To remove the row and column, work with indices and values
    indices = adj.coalesce().indices()
    # values = adj.coalesce().values()
    size = list(adj.size())

    src, dst = adj.coalesce().indices()

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)
    with torch.no_grad():
        pred_ori= model(g, features)
    ori_dist = avg_dist(group, pred_ori)

    # New size after removal
    # new_size = [size[0]-1, size[1]-1]

    # create an empty np array to store the individual bias contribution
    individual_bias = np.zeros(size[0])

    if device.type == 'cuda':
        total_memory = torch.cuda.get_device_properties(device).total_memory

    for i in tqdm(range(size[0])):
        allocated = torch.cuda.memory_allocated(device)  # Get allocated memory on the current device
        total_memory = torch.cuda.get_device_properties(device).total_memory
        memory_usage = allocated / total_memory
        
        if memory_usage > 0.9:
            current_device = (current_device + 1) % num_gpus  # Switch to the next GPU
            torch.cuda.set_device(current_device)  # Set the new device
            device = torch.device(f'cuda:{current_device}')  # Update the device
            model.to(device)  # Move model to the new device
            features = features.to(device)  # Move features to the new device

        if device.type == 'cuda':
            # allocated = torch.cuda.memory_allocated()
            print(allocated / total_memory)

        # start_time = time.time()
        # Remove indices corresponding to the ith row and column
        # mask_row = indices[0] != i
        # mask_col = indices[1] != i
        # mask = mask_row & mask_col
        # filtered_indices = indices[:, mask]
        filtered_indices = indices[:, (indices[0] != i) & (indices[1] != i)]
        # filtered_values = values[mask]
        # indices_adjustment_start_time = time.time()
        # Adjust indices to account for the removed row and column
        filtered_indices[0, filtered_indices[0] > i] -= 1
        filtered_indices[1, filtered_indices[1] > i] -= 1
        # print(f"Index adjustment time: {time.time() - indices_adjustment_start_time:.6f} seconds")
        # src = filtered_indices[0]
        # dst = filtered_indices[1]
        # graph_creation_start_time = time.time()
        # g_new = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
        # g_new = dgl.heterograph({('node', 'edge', 'node'): (src, dst)})
        # g_new = dgl.heterograph({('node', 'edge', 'node'): (filtered_indices[0], filtered_indices[1])})
        # g_new = g_new.int().to(device)
        # print(f"Graph creation time: {time.time() - graph_creation_start_time:.6f} seconds")

        # Remove the ith row and column from the feature matrix
        mask = torch.ones(size[0], dtype=bool)
        mask[i] = False
        # prediction_start_time = time.time()
        # Get the prediction for the new graph
        with torch.no_grad():
            # pred = model(g_new, features[mask])
            pred = model(dgl.heterograph({('node', 'edge', 'node'): (filtered_indices[0], filtered_indices[1])}).int().to(device), 
                        features[mask])
        # print(f"Prediction time: {time.time() - prediction_start_time:.6f} seconds")
        # remove the ith element from the group
        # bias_computation_start_time = time.time()
        # Compute the individual bias contribution
        avg_dist_tmp = avg_dist(group[mask], pred)
        # print(f"Bias computation time: {time.time() - bias_computation_start_time:.6f} seconds")
        del filtered_indices
        # del g_new
        del mask
        del pred
        torch.cuda.empty_cache()
        individual_bias[i] = avg_dist_tmp

    # move model back to gpu 0
    model.to(torch.device('cuda:0'))

    individual_bias = ori_dist - individual_bias
    return individual_bias


def group_bias_contribution(adj, features, model, group, selected_nodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # To remove the row and column, work with indices and values
    indices = adj.coalesce().indices()
    print(len(torch.unique(indices)))

    src, dst = adj.coalesce().indices()

    size = list(adj.size())

    # Create the heterograph
    g = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
    g = g.int().to(device)
    with torch.no_grad():
        pred_ori= model(g, features)
    ori_dist = avg_dist(group, pred_ori)

    # Remove indices corresponding to the ith row and column
    selected_nodes_tensor = torch.tensor(selected_nodes, dtype=torch.long)
    mask_row = ~torch.isin(indices[0], selected_nodes_tensor)
    mask_col = ~torch.isin(indices[1], selected_nodes_tensor) 
    mask = mask_row & mask_col

    filtered_indices = indices[:, mask]
    # filtered_values = values[mask]
    print(len(torch.unique(filtered_indices)))  
    # Adjust indices to account for the removed row and column
    for i in np.sort(selected_nodes)[:: -1]:
        filtered_indices[0, filtered_indices[0] > i] -= 1
        filtered_indices[1, filtered_indices[1] > i] -= 1 
    src = filtered_indices[0]
    dst = filtered_indices[1]
    print(len(torch.unique(filtered_indices)))
    print(len(selected_nodes))
    g_new = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())}) 
    g_new = g_new.int().to(device)

    # Remove the ith row and column from the feature matrix
    mask = torch.ones(size[0], dtype=bool)
    mask[selected_nodes] = False
    new_features = features[mask]

    # Get the prediction for the new graph
    with torch.no_grad():
        pred = model(g_new, new_features)

    # remove the ith element from the group
    group_new = group[mask]

    # Compute the group bias contribution
    avg_dist_tmp = avg_dist(group_new, pred)
    group_bias = ori_dist - avg_dist_tmp
    return group_bias


def avg_dist(group_new, pred):
    # group_unique = group_new.unique()
    # get the unique values of np array group_new
    group_unique = np.unique(group_new)
    # for each pair of groups, compute the distance between their predictions
    dists = []
    for i in range(len(group_unique)):
        for j in range(i+1, len(group_unique)):
            pred_i = pred[group_new == group_unique[i]].detach().cpu().numpy()
            pred_j = pred[group_new == group_unique[j]].detach().cpu().numpy()
            # compute the distance between the predictions
            dist = jsdist(pred_i, pred_j)
            dists.append(dist)

    # Compute the average distance
    avg_dist = np.mean(dists)
    return avg_dist


def jsdist(set1, set2):
    n_classes = set1.shape[1]
    n_set1 = set1.shape[0]
    n_set2 = set2.shape[0]
    if n_set1 <= n_classes * 30:
        sampled_set1t = set1.T
    else:
        sampled_set1t = set1[np.random.choice(set1.shape[0], n_classes * 30, replace=False)].T
    if n_set2 <= n_classes * 30:
        sampled_set2t = set2.T
    else:
        sampled_set2t = set2[np.random.choice(set2.shape[0], n_classes * 30, replace=False)].T
    # kde_time_start = time.time()
    # Estimate PDFs using KDE
    # kde1 = gaussian_kde(set1.T)
    # kde2 = gaussian_kde(set2.T)
    kde1 = gaussian_kde(sampled_set1t)
    kde2 = gaussian_kde(sampled_set2t)
    # print(f"KDE time: {time.time() - kde_time_start:.6f} seconds")
    # Assuming your KDE objects are kde1 and kde2, created from set1 and set2

    # Combine the datasets to cover the support of both distributions
    # combined_set = np.hstack([set1.T, set2.T])  # Combine along the feature axis
    combined_set = np.hstack([sampled_set1t, sampled_set2t])  # Combine along the feature axis
    # print('shape of combined set:', combined_set.shape)
    # pdf_time_start = time.time()
    # Evaluate the densities of both KDEs on the combined set
    pdf1 = kde1(combined_set)
    pdf2 = kde2(combined_set)
    # print(f"PDF time: {time.time() - pdf_time_start:.6f} seconds")
    # Normalize the densities to ensure they sum to 1 (like probabilities)
    pdf1 /= pdf1.sum()
    pdf2 /= pdf2.sum()
    # js_time_start = time.time()
    # Compute JS divergence
    js_dist = jensenshannon(pdf1, pdf2, base=2)
    # print(f"JS divergence time: {time.time() - js_time_start:.6f} seconds")
    return js_dist