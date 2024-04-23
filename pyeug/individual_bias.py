import numpy as np
# from scipy.stats import gaussian_kde
# from scipy.spatial.distance import jensenshannon
import torch
import torch.distributions as dist
import dgl
from tqdm import tqdm
import time
import math
# from torch.profiler import profile, record_function, ProfilerActivity
# from . import util


def group_bias_contribution(adj, features, model, group, selected_nodes):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # To remove the row and column, work with indices and values
    indices = adj.coalesce().indices()

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
    # Adjust indices to account for the removed row and column
    for i in np.sort(selected_nodes)[:: -1]:
        filtered_indices[0, filtered_indices[0] > i] -= 1
        filtered_indices[1, filtered_indices[1] > i] -= 1 
    src = filtered_indices[0]
    dst = filtered_indices[1]
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
            pred_i = pred[group_new == group_unique[i]]
            pred_j = pred[group_new == group_unique[j]]
            # compute the distance between the predictions
            dist = jsdist(pred_i, pred_j)
            dists.append(dist)

    # Compute the average distance
    avg_dist = np.mean(dists)
    return avg_dist


# def jsdist(set1, set2):
#     n_classes = set1.shape[1]
#     n_set1 = set1.shape[0]
#     n_set2 = set2.shape[0]
#     if n_set1 <= n_classes * 30:
#         sampled_set1t = set1.T
#     else:
#         sampled_set1t = set1[np.random.choice(set1.shape[0], n_classes * 30, replace=False)].T
#     if n_set2 <= n_classes * 30:
#         sampled_set2t = set2.T
#     else:
#         sampled_set2t = set2[np.random.choice(set2.shape[0], n_classes * 30, replace=False)].T
#     # kde_time_start = time.time()
#     # Estimate PDFs using KDE
#     # kde1 = gaussian_kde(set1.T)
#     # kde2 = gaussian_kde(set2.T)
#     kde1 = gaussian_kde(set1.T) 
#     kde2 = gaussian_kde(set2.T) 
#     # print(f"KDE time: {time.time() - kde_time_start:.6f} seconds")
#     # Assuming your KDE objects are kde1 and kde2, created from set1 and set2

#     # Combine the datasets to cover the support of both distributions
#     combined_set = np.hstack([set1.T, set2.T])  # Combine along the feature axis
#     # combined_set = np.hstack([sampled_set1t, sampled_set2t])  # Combine along the feature axis
#     # print('shape of combined set:', combined_set.shape)
#     # pdf_time_start = time.time()
#     # Evaluate the densities of both KDEs on the combined set
#     pdf1 = kde1(combined_set)
#     pdf2 = kde2(combined_set)
#     # print(f"PDF time: {time.time() - pdf_time_start:.6f} seconds")
#     # Normalize the densities to ensure they sum to 1 (like probabilities)
#     pdf1 /= pdf1.sum()
#     pdf2 /= pdf2.sum()
#     # js_time_start = time.time()
#     # Compute JS divergence
#     js_dist = jensenshannon(pdf1, pdf2, base=2)
#     # print(f"JS divergence time: {time.time() - js_time_start:.6f} seconds")
#     return js_dist


def gaussian_kde_pytorch(data, bw_method='scott'):
    """
    Vectorized Gaussian KDE in PyTorch for multi-dimensional data.
    """
    n, d = data.size()
    if bw_method == 'scott':
        bandwidth = n ** (-1. / (d + 4))  # Scott's rule applied to each dimension

    # Creating a multivariate normal distribution with diagonal covariance
    scale = bandwidth * torch.eye(d, device=data.device)
    kernel = dist.MultivariateNormal(torch.zeros(d, device=data.device), scale_tril=scale)

    def evaluate(points):
        # Shape of points: [D, P] where D is dimensions and P is number of points
        # We need to compute the density at each point in `points` for each sample in `data`
        # Expand data to [P, N, D]
        # Expand points to [P, 1, D]
        points = points.t().unsqueeze(1)  # Shape: [P, 1, D]
        data0 = data.unsqueeze(0)  # Shape: [1, N, D]
        diffs = points - data0  # Broadcasting to get differences [P, N, D]
        # log_probs = kernel.log_prob(diffs)  # Sum over dimensions automatically, Shape: [P, N]
        log_probs = mvnlogprob(kernel, diffs)
        weights = torch.logsumexp(log_probs, dim=1) - torch.log(torch.tensor(n, dtype=torch.float32, device=data.device))
        return torch.exp(weights)  # Convert log probabilities back to probabilities

    return evaluate


def jsdist(set1, set2):
    kde1 = gaussian_kde_pytorch(set1)
    kde2 = gaussian_kde_pytorch(set2)

    combined_set = torch.cat([set1, set2], dim=0).t()
    pdf1 = kde1(combined_set)
    pdf2 = kde2(combined_set)

    pdf1 /= pdf1.sum()
    pdf2 /= pdf2.sum()

    m = 0.5 * (pdf1 + pdf2)
    kl1 = torch.sum(pdf1 * torch.log(pdf1 / m))
    kl2 = torch.sum(pdf2 * torch.log(pdf2 / m))
    js_dist = torch.sqrt(0.5 * (kl1 + kl2))

    return js_dist.item()

def mvnlogprob(dist: torch.distributions.multivariate_normal.MultivariateNormal, 
               inputs: torch.tensor):
    p = dist.loc.size(0)
    diff = inputs - dist.loc
    
    batch_shape = diff.shape[:-1]
    
    scale_shape = dist.scale_tril.size()
    
    _scale_tril = dist.scale_tril.expand(batch_shape+scale_shape)
    z = torch.linalg.solve_triangular(_scale_tril,
                                      diff.unsqueeze(-1), 
                                      upper=False).squeeze()
    
    out=  -0.5*p*torch.tensor(2*math.pi).log() - dist.scale_tril.logdet() -0.5*(z**2).sum(dim=-1)
    return out.squeeze()