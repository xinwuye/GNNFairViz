import numpy as np
import torch
import torch.distributions as dist
import dgl
import math

SEED = 42
np.random.seed(SEED)


def calc_attr_contributions(model, g, feat, selected_nodes, groups, attr_indices):
    feat_mean = feat.mean(dim=tuple(range(feat.dim() - 1)))

    # Set the model to evaluation
    model.eval()
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(g, feat)
    ori_dist = avg_dist(groups, embeddings) 

    feat_clone = feat.clone().detach()
    # Indices for rows and columns where the values need to be set to 1
    row_indices = torch.tensor(selected_nodes.astype(int))  # Rows 1 and 3
    col_indices = torch.tensor(attr_indices)  # Columns 0 and 2

    # Convert row and column indices to a meshgrid of indices
    rows, cols = torch.meshgrid(row_indices, col_indices, indexing='ij')

    # Use fancy indexing to set the specified elements to 1
    feat_clone[rows, cols] = feat_mean[attr_indices] 

    with torch.no_grad():
        embeddings_perturbed = model(g, feat_clone)
    dist_perturbed = avg_dist(groups, embeddings_perturbed) 
    contributions = (ori_dist - dist_perturbed) / ori_dist 
    return contributions


def calc_structure_contribution(adj, model, g, feat, selected_nodes, groups):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Set the model to evaluation
    model.eval()
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(g, feat)

    ori_dist = avg_dist(groups, embeddings) 

    # remove the ith row and column from the adjacency matrix
    adj_clone = adj.clone().detach()

    # Converting the rows_to_zero into a tensor
    selected_nodes_tensor = torch.from_numpy(selected_nodes.astype(int)).to(device)

    # Get the row indices and column indices separately from the adj tensor
    indices = adj_clone.coalesce().indices()
    row_indices = indices[0]
    col_indices = indices[1]

    # Mask to zero out the selected rows
    row_mask = ~torch.isin(row_indices, selected_nodes_tensor)
    col_mask = ~torch.isin(col_indices, selected_nodes_tensor)
    mask = row_mask & col_mask

    # Apply mask
    new_row_indices = row_indices[mask]
    new_col_indices = col_indices[mask]

    # Determine which diagonal indices need to be added
    needed_diagonals = selected_nodes_tensor
    existing_diagonals = (new_row_indices == new_col_indices) & torch.isin(new_row_indices, needed_diagonals)
    missing_diagonals = needed_diagonals[~torch.isin(needed_diagonals, new_row_indices[existing_diagonals])]

    # Add missing diagonal elements
    if missing_diagonals.numel() > 0:
        new_row_indices = torch.cat([new_row_indices, missing_diagonals])
        new_col_indices = torch.cat([new_col_indices, missing_diagonals])

    g_new = dgl.heterograph({('node', 'edge', 'node'): (new_row_indices.cpu().numpy(), new_col_indices.cpu().numpy())}) 
    g_new = g_new.int().to(feat.device)

    # Get the prediction for the new graph
    with torch.no_grad():
        embeddings_new = model(g_new, feat)

    new_dist = avg_dist(groups, embeddings_new) 

    contributions = (ori_dist - new_dist) / ori_dist 

    return contributions


def calc_attr_structure_contribution(adj, model, g, feat, selected_nodes, groups, attr_indices):
    feat_mean = feat.mean(dim=tuple(range(feat.dim() - 1)))

    # Set the model to evaluation
    model.eval()
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(g, feat)
    ori_dist = avg_dist(groups, embeddings) 

    # remove the ith row and column from the adjacency matrix
    adj_clone = adj.clone().detach()

    # Converting the rows_to_zero into a tensor
    selected_nodes_tensor = torch.from_numpy(selected_nodes.astype(int)).to(feat.device)

    # Get the row indices and column indices separately from the adj tensor
    indices = adj_clone.coalesce().indices()
    row_indices = indices[0]
    col_indices = indices[1]

    # Mask to zero out the selected rows
    row_mask = ~torch.isin(row_indices, selected_nodes_tensor)
    col_mask = ~torch.isin(col_indices, selected_nodes_tensor)
    mask = row_mask & col_mask

    # Apply mask
    new_row_indices = row_indices[mask]
    new_col_indices = col_indices[mask]

    # Determine which diagonal indices need to be added
    needed_diagonals = selected_nodes_tensor
    existing_diagonals = (new_row_indices == new_col_indices) & torch.isin(new_row_indices, needed_diagonals)
    missing_diagonals = needed_diagonals[~torch.isin(needed_diagonals, new_row_indices[existing_diagonals])]

    # Add missing diagonal elements
    if missing_diagonals.numel() > 0:
        new_row_indices = torch.cat([new_row_indices, missing_diagonals])
        new_col_indices = torch.cat([new_col_indices, missing_diagonals])

    g_new = dgl.heterograph({('node', 'edge', 'node'): (new_row_indices.cpu().numpy(), new_col_indices.cpu().numpy())}) 
    g_new = g_new.int().to(feat.device)

    # Remove the ith row and column from the feature matrix
    feat_clone = feat.clone().detach()
    # Indices for rows and columns where the values need
    row_indices = torch.tensor(selected_nodes.astype(int))  # Rows 1 and 3
    col_indices = torch.tensor(attr_indices)  # Columns 0 and 2

    # Convert row and column indices to a meshgrid of indices
    rows, cols = torch.meshgrid(row_indices, col_indices, indexing='ij')

    # Use fancy indexing to set the specified elements to 1
    feat_clone[rows, cols] = feat_mean[attr_indices]

    with torch.no_grad():
        embeddings_new = model(g_new, feat_clone)
    new_dist = avg_dist(groups, embeddings_new)

    contributions = (ori_dist - new_dist) / ori_dist

    return contributions


def avg_dist(group_new, pred):
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

    batch_size = 10**8 // n  # Maximum number of points that can be processed at once

    def evaluate(points):
        # Ensure points are in the correct shape [D, P]
        points = points.t()  # Transpose points if necessary
        data0 = data.unsqueeze(0)  # Shape: [1, N, D]
        total_points = points.shape[0]
        densities = torch.zeros(total_points, device=data.device)

        # Process points in batches
        for i in range(0, total_points, batch_size):
            batch_points = points[i:i + batch_size].unsqueeze(1)  # Reshape for broadcasting
            diffs = batch_points - data0  # Compute differences for this batch
            log_probs = mvnlogprob(kernel, diffs)
            weights = torch.logsumexp(log_probs, dim=1) - torch.log(torch.tensor(n, dtype=torch.float32, device=data.device))
            densities[i:i + batch_size] = torch.exp(weights)  # Store results in the densities tensor

        return densities

    return evaluate


def jsdist(set1, set2):
    kde1 = gaussian_kde_pytorch(set1)
    kde2 = gaussian_kde_pytorch(set2)

    combined_set = torch.cat([set1, set2], dim=0).t()
    pdf1 = kde1(combined_set)
    pdf2 = kde2(combined_set)
    pos_indices = (pdf1 > 0) & (pdf2 > 0)
    pdf1 = pdf1[pos_indices]
    pdf2 = pdf2[pos_indices]

    pdf1 /= pdf1.sum()
    pdf2 /= pdf2.sum()

    m = 0.5 * (pdf1 + pdf2)

    kl1 = torch.sum(pdf1 * torch.log(pdf1 / m))
    kl2 = torch.sum(pdf2 * torch.log(pdf2 / m))

    js_dist = torch.sqrt(0.5 * (kl1 + kl2))

    if torch.isnan(js_dist):
        return 0
    else:
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