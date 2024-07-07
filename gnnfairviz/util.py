import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from scipy.stats import kruskal, mannwhitneyu, chi2_contingency
import torch
from tqdm import tqdm
from . import individual_bias


torch.manual_seed(42)


def proj_emb(embeddings_ls, method_name, perplexity=15, seed=42): 
    '''
    Project embeddings to 2D space
    Args:
        embeddings_ls: list of embeddings
        projector: sklearn.manifold.TSNE or sklearn.decomposition.PCA
    Returns:
        embeddings_2d_ls: list of 2D embeddings
    '''  
    embeddings_2d_ls = []
    if method_name == 'tsne':
        projector = TSNE(perplexity=perplexity, 
                n_components=2, 
                init='pca', 
                n_iter=1250, 
                learning_rate='auto', random_state=seed)
        for e in embeddings_ls:
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
                embeddings_2d = projector.fit_transform(e)
                embeddings_2d_ls.append(embeddings_2d)
    elif method_name == 'pca':
        projector = PCA(n_components=2)
        for e in embeddings_ls:
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
                embeddings_2d = projector.fit_transform(e)
                embeddings_2d_ls.append(embeddings_2d)
    elif method_name == 'umap':
        projector = umap.UMAP(n_components=2, random_state=seed)
        for e in embeddings_ls:
            if isinstance(e, torch.Tensor):
                e = e.detach().cpu().numpy()
                embeddings_2d = projector.fit_transform(e)
                embeddings_2d_ls.append(embeddings_2d)
    return embeddings_2d_ls


def init(model, feat, g, emb_fnc):
    embeddings = emb_fnc(g, feat, model)
    embeddings = [e.detach().clone() for e in embeddings]

    return {'embeddings': embeddings,
            }


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


# Function to modify the sparse tensor using SciPy
def modify_sparse_tensor_scipy(adj):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    adj = adj.clone().coalesce()  # Ensure indices are sorted and duplicate indices are combined
    # Convert PyTorch sparse tensor to SciPy COO matrix
    indices = adj.indices().numpy()
    values = adj.values().numpy()
    shape = adj.size()
    scipy_adj_coo = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape)
    scipy_adj = scipy_adj_coo.tocsr()
    # deepcopy scipy_adj
    modified_scipy_adj = scipy_adj.copy()

    # Check if [0, 0] is 1
    if scipy_adj[0, 0] == 1:
        # Filter out diagonal elements
        rows, cols = scipy_adj_coo.row, scipy_adj_coo.col
        data = scipy_adj_coo.data
        non_diagonal_indices = rows != cols

        filtered_rows = rows[non_diagonal_indices]
        filtered_cols = cols[non_diagonal_indices]
        filtered_data = data[non_diagonal_indices]

        # Create a new CSR matrix without diagonal elements (or explicit zeros on the diagonal)
        modified_scipy_adj = csr_matrix((filtered_data, (filtered_rows, filtered_cols)), shape=scipy_adj.shape)
        # to coo
        modified_scipy_adj_coo = modified_scipy_adj.tocoo()

        # Instead of directly converting the list of numpy arrays to a tensor,
        # first, convert the list of numpy arrays (rows and cols) to a single numpy array.
        rows_cols_combined = np.array([modified_scipy_adj_coo.row, modified_scipy_adj_coo.col])

        # Then, use this numpy array to create the PyTorch tensor.
        modified_adj = torch.sparse_coo_tensor(torch.LongTensor(rows_cols_combined), 
                                            torch.FloatTensor(modified_scipy_adj_coo.data), modified_scipy_adj_coo.shape)
        return modified_adj.to(device), adj, modified_scipy_adj, scipy_adj
    else:
        modified_scipy_adj.setdiag(1)
        # to coo
        modified_scipy_adj_coo = modified_scipy_adj.tocoo()

        # Instead of directly converting the list of numpy arrays to a tensor,
        # first, convert the list of numpy arrays (rows and cols) to a single numpy array.
        rows_cols_combined = np.array([modified_scipy_adj_coo.row, modified_scipy_adj_coo.col])

        # Then, use this numpy array to create the PyTorch tensor.
        modified_adj = torch.sparse_coo_tensor(torch.LongTensor(rows_cols_combined), 
                                            torch.FloatTensor(modified_scipy_adj_coo.data), modified_scipy_adj_coo.shape)
        return adj, modified_adj.to(device), scipy_adj, modified_scipy_adj


def calculate_graph_metrics(adj_matrices):
    """
    Calculate the size and density of each graph represented by adjacency matrices.

    :param adj_matrices: List of adjacency matrices in CSR format
    :return: List of tuples containing size and density of each graph
    """
    metrics = []

    for adj_matrix in adj_matrices:
        # Number of edges: sum of non-zero elements in the matrix
        num_edges = adj_matrix.nnz // 2  # Dividing by 2 for undirected graph

        # Number of nodes: dimension of the matrix
        num_nodes = adj_matrix.shape[0]

        # Maximum possible number of edges in an undirected graph
        max_edges = num_nodes * (num_nodes - 1) // 2

        # Density: Actual edges / Maximum possible edges
        density = num_edges / max_edges if max_edges > 0 else 0
        metrics.append((num_nodes, density))

    return metrics


def analyze_bias(feat, groups, columns_categorical, selected_nodes):
    m = feat.shape[-1]

    # Identify all unique groups in the dataset
    all_value_counts = groups.value_counts()
    all_unique_groups = all_value_counts.index
    num_all_groups = len(all_unique_groups)
    all_ns = all_value_counts.values

    # feat as a DataFrame, slice the rows with selected_nodes as index
    feat_selected = feat.iloc[selected_nodes]
    groups_selected = groups.iloc[selected_nodes]
    value_counts = groups_selected.value_counts()
    selected_unique_groups = value_counts.index
    num_groups = len(selected_unique_groups)

    # Initialize ns for all groups with 0s
    ns = np.zeros(num_all_groups, dtype=int)
    for group_name in selected_unique_groups:
        group_index = np.where(all_unique_groups == group_name)[0][0]
        ns[group_index] = value_counts[group_name]

    # Initialize bias_indicator with None for all features and groups
    bias_indicator = np.zeros((m, num_all_groups), dtype=bool)  # Adjusted dimensions
    overall_bias_indicator = np.zeros(m, dtype=bool)

    if num_groups > 1:
        for i in range(m):
            variable_data = feat_selected.iloc[:, i]
            if columns_categorical[i]:  # If the column is categorical
                contingency_table = pd.crosstab(variable_data, groups_selected)
                chi2_stat, chi2_p, dof, expected = chi2_contingency(contingency_table)
                if chi2_p <= 0.05:
                    overall_bias_indicator[i] = True

                for j, group_name in enumerate(all_unique_groups):
                    if group_name in selected_unique_groups:
                        group_data = contingency_table.get(group_name, pd.Series([]))
                        other_data = contingency_table.drop(columns=group_name, errors='ignore')
                        stat, p, dof, expected = chi2_contingency(pd.concat([group_data, other_data], axis=1))
                        bias_indicator[i, j] = p <= 0.05
            else:
                df = pd.concat([variable_data, groups_selected], axis=1, keys=['Data', 'Group'])
                # check if all the values in the Data col is the same
                if df['Data'].nunique() > 1:
                    kruskal_stat, kruskal_p = kruskal(*[group['Data'].values for name, group in df.groupby('Group')])
                else:
                    kruskal_p = 1.0
                if kruskal_p <= 0.05:
                    overall_bias_indicator[i] = True

                for j, group_name in enumerate(all_unique_groups):
                    if group_name in selected_unique_groups:
                        group_data = df[df['Group'] == group_name]['Data']
                        other_data = df[df['Group'] != group_name]['Data']
                        stat, p = mannwhitneyu(group_data, other_data, alternative='two-sided')
                        bias_indicator[i, j] = p <= 0.05

    return bias_indicator, overall_bias_indicator, ns, all_ns, all_unique_groups


def calc_contributions(model, g, feat, selected_nodes, groups):
    n_feat = feat.shape[-1]
    feat_mean = feat.mean(dim=tuple(range(feat.dim() - 1)))

    # Set the model to evaluation
    model.eval()
    # Get the embeddings
    with torch.no_grad():
        embeddings = model(g, feat)
    ori_dist = individual_bias.avg_dist(groups, embeddings)

    # for each column in feat
    contributions = np.zeros(n_feat)
    # for i in tqdm(range(n_feat)):
    for i in range(n_feat):
        feat_clone = feat.clone().detach()
        feat_clone[..., selected_nodes, i] = feat_mean[i]
        with torch.no_grad():
            embeddings_perturbed = model(g, feat_clone)
        dist_perturbed = individual_bias.avg_dist(groups, embeddings_perturbed)
        contributions[i] = dist_perturbed
    contributions = (ori_dist - contributions) / ori_dist 
    return contributions

