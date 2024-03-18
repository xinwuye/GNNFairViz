import numpy as np
from scipy.stats import gaussian_kde
from scipy.spatial.distance import jensenshannon
import torch
import dgl

def individual_bias_contribution(adj, features, model, group):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

    for i in range(size[0]):
        # Remove indices corresponding to the ith row and column
        mask_row = indices[0] != i
        mask_col = indices[1] != i
        mask = mask_row & mask_col

        filtered_indices = indices[:, mask]
        # filtered_values = values[mask]

        # Adjust indices to account for the removed row and column
        filtered_indices[0, filtered_indices[0] > i] -= 1
        filtered_indices[1, filtered_indices[1] > i] -= 1
        src = filtered_indices[0]
        dst = filtered_indices[1]
        g_new = dgl.heterograph({('node', 'edge', 'node'): (src.cpu().numpy(), dst.cpu().numpy())})
        g_new = g_new.int().to(device)

        # Remove the ith row and column from the feature matrix
        mask = torch.ones(size[0], dtype=bool)
        mask[i] = False
        new_features = features[mask]

        # Get the prediction for the new graph
        with torch.no_grad():
            pred = model(g_new, new_features)

        # remove the ith element from the group
        group_new = group[mask]

        # Compute the individual bias contribution
        avg_dist_tmp = avg_dist(group_new, pred)
        individual_bias[i] = avg_dist_tmp

    individual_bias = ori_dist - individual_bias
    return individual_bias


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
    # Estimate PDFs using KDE
    kde1 = gaussian_kde(set1.T)
    kde2 = gaussian_kde(set2.T)

    # Assuming your KDE objects are kde1 and kde2, created from set1 and set2

    # Combine the datasets to cover the support of both distributions
    combined_set = np.hstack([set1.T, set2.T])  # Combine along the feature axis

    # Evaluate the densities of both KDEs on the combined set
    pdf1 = kde1(combined_set)
    pdf2 = kde2(combined_set)

    # Normalize the densities to ensure they sum to 1 (like probabilities)
    pdf1 /= pdf1.sum()
    pdf2 /= pdf2.sum()

    # Compute JS divergence
    js_dist = jensenshannon(pdf1, pdf2, base=2)

    return js_dist