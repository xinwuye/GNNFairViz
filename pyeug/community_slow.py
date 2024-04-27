import numpy as np
import networkx as nx
from scipy.spatial import distance
import torch
import scipy.sparse as sp
from tqdm import tqdm
import cProfile
import pstats
import sys


# Structure for Nodes of Dendogram
class Node:
    def __init__(self, name):
        self.name = name
        self.left = None
        self.right = None
        self.parent = None
        self.root = self
        self.num_edges = 0
        self.vertices = set()
        self.density = 0


# Tree structure to handle dendogram operations
class Tree:
    def __init__(self):
        self.root = None
        self.communities = []

    # Find Lowest Common Ancestor
    def findLCA_Node(self, src_node, dest_node):
        while src_node is not None:
            if dest_node.name in src_node.vertices:
                return src_node
            src_node = src_node.parent
        return None

    # def print_Tree(self, root):
    #     if root is None:
    #         return
    #     print(root.vertices)
    #     self.print_Tree(root.left)
    #     self.print_Tree(root.right)

    # def print_nodes(self, nodes):
    #     for node in nodes:
    #         print(node.name, node.parent.name if node.parent else None)

    def count_vertices_and_edges(self, edges_list, nodes_list):
        for edge in edges_list:
            lca_node = None
            src_node = nodes_list.get(edge[0])
            dst_node = nodes_list.get(edge[1])
            if src_node and dst_node:
                lca_node = self.findLCA_Node(src_node, dst_node)
            if lca_node:
                lca_node.num_edges += 1
            
    # def count_vertices_and_edges(self, adj, nodes_list):
    #     indices = adj.coalesce().indices()
    #     # for edge in edges_list:
    #     for i in range(indices.size(1)):
    #         edge = indices[:, i].tolist()
    #         lca_node = None
    #         src_node = nodes_list.get(edge[0])
    #         dst_node = nodes_list.get(edge[1])
    #         if src_node and dst_node:
    #             lca_node = self.findLCA_Node(src_node, dst_node)
    #         if lca_node:
    #             lca_node.num_edges += 1

    def count_vertices_and_edges_wrap(self, root):
        if root.left and root.right:
            self.count_vertices_and_edges_wrap(root.left)
            self.count_vertices_and_edges_wrap(root.right)
            root.num_edges = root.left.num_edges + root.right.num_edges + root.num_edges

    def compute_density(self, root):
        if root.left is None and root.right is None:
            return
        total_vertices = float(len(root.vertices))
        max_vertices = total_vertices * (total_vertices - 1) / 2
        root.density = root.num_edges / max_vertices if max_vertices else 0
        self.compute_density(root.left)
        self.compute_density(root.right)

    def extract_sub_graph(self, root, min_density):
        if root is None:
            return
        if root.density > min_density:
            self.communities.append(root.vertices)
        else:
            self.extract_sub_graph(root.left, min_density)
            self.extract_sub_graph(root.right, min_density)


def MakeSet(node):
    # node.parent = None
    node.vertices.add(node.name)


# def SetFind(node):
#     while node.parent:
#         node = node.parent
#     return node
def SetFind(node):
    if node.root is not node:
        node.root = SetFind(node.root)  # Path compression
    return node.root


def SetUnion(x, y):
    r = Node(f"P{x.name}{y.name}")
    r.left = x
    r.right = y
    x.parent = r
    y.parent = r
    x.root = r
    y.root = r
    r.vertices = r.vertices.union(x.vertices, y.vertices)
    return r


def create_networkx_graph_from_sparse_tensor(sparse_tensor):
    # Initialize an empty graph
    G = nx.Graph()

    # Add nodes to the graph
    n_nodes = sparse_tensor.size(0)
    G.add_nodes_from(range(n_nodes))

    # Extract indices from the sparse tensor
    indices = sparse_tensor._indices()

    # Iterate over the edges and add them to the graph
    for i in range(indices.size(1)):
        n1, n2 = indices[:, i].tolist()
        G.add_edge(n1, n2)

    return G


def process_graph(adj):
    G = create_networkx_graph_from_sparse_tensor(adj)
    normed_adj = normalize_sparse_tensor_by_row(adj)
    M = extract_upper_triangular(torch.sparse.mm(normed_adj, normed_adj.T)).coalesce() 
    nnz = adj._nnz() 
    print('nnz', nnz)

    # top_values, top_indices = find_top_k_sparse(M, nnz)
    top_indices = find_top_k_sparse(M, nnz).T 

    nodes =dict() 
    root_nodes = set()

    # profiler = cProfile.Profile()
    # profiler.enable()  # Start profiling
    # for i, value in enumerate(top_values):
    for vertices in tqdm(top_indices):
        # vertices = top_indices[:, i]
        i,j = vertices
        if nodes.__contains__(i) is False:
            a = Node(i)
            MakeSet(a)
            nodes[i] = a
        if nodes.__contains__(j) is False:
            a = Node(j)
            MakeSet(a)
            nodes[j]=a
		
        i = nodes[i]
        j = nodes[j]
        ri = SetFind(i)
        rj = SetFind(j)
        if ri.vertices != rj.vertices:
            temp_root = SetUnion(ri,rj)
            root_nodes.add(temp_root)
    # profiler.disable()  # Stop profiling
    # stats = pstats.Stats(profiler)
    # stats.strip_dirs().sort_stats('time').print_stats()
    # stats.print_callers()  # Optionally, to see who is calling what

    root_nodes = list(filter( lambda entry: entry.parent==None, list(root_nodes)))
    
    return Tree(), root_nodes, nodes, G


def extract_communities(tree, root_nodes, nodes, G, min_threshold):
    tree.communities = []
    #Counting number of vertices and Edges
    tree.count_vertices_and_edges(G.edges(),nodes)
    # for temp_roots in tqdm(root_nodes):
    for temp_roots in root_nodes:

        tree.root = temp_roots
		# #Counting number of vertices and Edges
        # tree.count_vertices_and_edges(G.edges(),nodes)
		#Summing up number of edges of children to parent
        tree.count_vertices_and_edges_wrap(tree.root)
		#Computing density of Tree Nodes
        tree.compute_density(tree.root)
		#Filtering Nodes as Per Density Threshold
        tree.extract_sub_graph(tree.root, min_threshold)

    return tree.communities


# def extract_communities(tree, root_nodes, nodes, adj, min_threshold):
#     G = create_networkx_graph_from_sparse_tensor(adj)
#     tree.communities = []
#     # for temp_roots in tqdm(root_nodes):
#     for temp_roots in tqdm(root_nodes):

#         tree.root = temp_roots
# 		#Counting number of vertices and Edges
#         # tree.count_vertices_and_edges(adj,nodes)
#         tree.count_vertices_and_edges(G.edges(),nodes)
# 		#Summing up number of edges of children to parent
#         tree.count_vertices_and_edges_wrap(tree.root)
# 		#Computing density of Tree Nodes
#         tree.compute_density(tree.root)
# 		#Filtering Nodes as Per Density Threshold
#         tree.extract_sub_graph(tree.root, min_threshold)

#     return tree.communities


def create_community_slices(sparse_adj_matrix, communities):
    # Convert PyTorch sparse tensor to Scipy sparse matrix (CSR format)
    sparse_adj_matrix = sparse_adj_matrix.coalesce()  # Ensures row and column indices are sorted
    indices = sparse_adj_matrix.indices().numpy()
    values = sparse_adj_matrix.values().numpy()
    shape = sparse_adj_matrix.shape
    scipy_sparse_matrix = sp.coo_matrix((values, (indices[0], indices[1])), shape=shape).tocsr()

    # Initialize a list to store the community adjacency matrices
    community_adj_matrices = []

    # Iterate over each community
    for community in communities:
        # Use advanced indexing to extract the submatrix
        community = list(community)
        community_submatrix = scipy_sparse_matrix[community, :][:, community].todense()
        community_adj_matrices.append(community_submatrix)

    return community_adj_matrices


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
    print(values)   
    
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