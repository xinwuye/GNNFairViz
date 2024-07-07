import torch
from tqdm import tqdm


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Structure for Nodes of Dendogram
class Node:
    def __init__(self, name):
        self.name = name
        self.left = None
        self.right = None
        self.parent = None
        self.root = self
        self.density = 0
        self.vertices = [self.name]


def extract_sub_graph(root, min_density):
    communities = []
    if root is None:
        return []
    if root.density > min_density:
        communities.append(root.vertices) # add vertices of this community
    else:
        communities.extend(extract_sub_graph(root.left, min_density))
        communities.extend(extract_sub_graph(root.right, min_density))
    return communities


def SetFind(node):
    if node.root is not node:
        node.root = SetFind(node.root)  # Path compression
    return node.root


def SetUnion(x, y, adj):
    r = Node(f"P{x.name}{y.name}")
    r.left = x
    r.right = y
    x.parent = r
    y.parent = r
    x.root = r
    y.root = r
    r.vertices = x.vertices + y.vertices
    selected_rows = torch.index_select(adj, 0, torch.tensor(r.vertices).to(device))
    selected_block = torch.index_select(selected_rows, 1, torch.tensor(r.vertices).to(device))
    n = len(r.vertices)
    r.density = selected_block._nnz() / (n * (n-1))
    return r


def process_graph(adj):
    normed_adj = normalize_sparse_tensor_by_row(adj)
    M = extract_upper_triangular(torch.sparse.mm(normed_adj, normed_adj.T)).coalesce() 
    nnz = adj._nnz() 

    top_indices = find_top_k_sparse(M, nnz).T 

    nodes =dict() 
    root_nodes = set()

    # for i, value in enumerate(top_values):
    # for vertices in tqdm(top_indices):
    for vertices in top_indices:
        i,j = vertices
        if nodes.__contains__(i) is False:
            a = Node(i)
            nodes[i] = a
        if nodes.__contains__(j) is False:
            a = Node(j)
            nodes[j]=a
		
        i = nodes[i]
        j = nodes[j]
        ri = SetFind(i)
        rj = SetFind(j)
        if ri != rj:
            temp_root = SetUnion(ri,rj, adj)
            root_nodes.add(temp_root)

    root_nodes = list(filter(lambda entry: entry.parent==None, list(root_nodes)))
    
    return root_nodes


def extract_communities(root_nodes, min_threshold):
    communities = []
    # for temp_root in tqdm(root_nodes):
    for temp_root in root_nodes:
		#Filtering Nodes as Per Density Threshold
        communities_tmp = extract_sub_graph(temp_root, min_threshold)
        communities.extend(communities_tmp)

    return communities


def find_top_k_sparse(sparse_tensor, k):
    # Retrieve the values and the indices from the sparse tensor
    values = sparse_tensor.values()
    indices = sparse_tensor.indices()

    # Find the top k values and their indices
    top_values, top_positions = torch.topk(values, k, largest=True, sorted=True)

    # Extract the corresponding indices for the top values
    top_indices = indices[:, top_positions]

    # return top_values.flip(0).cpu().numpy(), top_indices.flip(1).cpu().numpy()
    return top_indices.cpu().numpy()   


def extract_upper_triangular(sparse_tensor):
    # Access indices and values of the sparse tensor
    indices = sparse_tensor.indices()
    values = sparse_tensor.values()
    
    # Find elements where the column index is greater than or equal to the row index
    mask = indices[1] >= indices[0]
    
    # Apply mask to filter indices and values
    filtered_indices = indices[:, mask]
    filtered_values = values[mask]
    
    # Create a new sparse tensor with the filtered indices and values
    upper_triangular_tensor = torch.sparse_coo_tensor(filtered_indices, filtered_values, sparse_tensor.shape, device=sparse_tensor.device)
    
    return upper_triangular_tensor


def normalize_sparse_tensor_by_row(sparse_tensor):
    # Calculate row sums
    row_sums = torch.sparse.sum(sparse_tensor, dim=1)

    # Convert row sums to dense tensor and take reciprocals (avoiding division by zero)
    row_sums_dense = torch.sqrt(row_sums.to_dense())
    row_reciprocals = 1.0 / row_sums_dense
    row_reciprocals[torch.isinf(row_reciprocals)] = 0  # Set infinities to zero (where row sums were zero)

    # Create indices for the diagonal matrix
    indices = torch.arange(sparse_tensor.shape[0], device=sparse_tensor.device)
    diag_indices = torch.stack([indices, indices])

    # Create a diagonal matrix in sparse format
    diag_matrix = torch.sparse_coo_tensor(diag_indices, row_reciprocals, sparse_tensor.shape, device=sparse_tensor.device)

    # Normalize the sparse matrix by row via matrix multiplication
    normalized_sparse_tensor = torch.sparse.mm(diag_matrix, sparse_tensor)

    return normalized_sparse_tensor

