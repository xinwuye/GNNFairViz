import numpy as np
import networkx as nx
from scipy.spatial import distance
import torch
import scipy.sparse as sp
from tqdm import tqdm


# Structure for Nodes of Dendogram
class Node:
    def __init__(self, name):
        self.name = name
        self.left = None
        self.right = None
        self.parent = None
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

    def print_Tree(self, root):
        if root is None:
            return
        print(root.vertices)
        self.print_Tree(root.left)
        self.print_Tree(root.right)

    def print_nodes(self, nodes):
        for node in nodes:
            print(node.name, node.parent.name if node.parent else None)

    def count_vertices_and_edges(self, edges_list, nodes_list):
        for edge in edges_list:
            lca_node = None
            src_node = nodes_list.get(edge[0])
            dst_node = nodes_list.get(edge[1])
            if src_node and dst_node:
                lca_node = self.findLCA_Node(src_node, dst_node)
            if lca_node:
                lca_node.num_edges += 1

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
        # print(root.vertices, root.density)
        if root.density > min_density:
            self.communities.append(root.vertices)
            # print("Community Detected:", ' '.join(map(str, root.vertices)))
        else:
            self.extract_sub_graph(root.left, min_density)
            self.extract_sub_graph(root.right, min_density)


def MakeSet(node):
    node.parent = None
    node.vertices.add(node.name)


def SetFind(node):
    while node.parent:
        node = node.parent
    return node


def SetUnion(x, y):
    r = Node(f"P{x.name}{y.name}")
    r.left = x
    r.right = y
    x.parent = r
    y.parent = r
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


# Function to process the graph file and build dendogram
def process_graph(adj):
    G = create_networkx_graph_from_sparse_tensor(adj)

    A = nx.adjacency_matrix(G)
    adj_matrix = A.todense()
    print(adj_matrix)

    M = np.zeros(adj_matrix.shape)
    row, col = adj_matrix.shape

    # Building similarity function matrix (Cosine Function matrix of all Column Vectors)
    # for x in tqdm(range(row)):
    for x in range(row):
        for y in range(x, col):
            if np.count_nonzero(adj_matrix[:, x]) and np.count_nonzero(adj_matrix[:, y]):
                M[x, y] = 1 - distance.cosine(adj_matrix[:, x], adj_matrix[:, y])

    tuples = []
	#On basis of zero graph
    # min_value = 1 if min(vertices)>0 else 0
    min_value = 0
	#Considering only non zero values
    # for (x,y), value in tqdm(np.ndenumerate(M)):
    for (x, y), value in np.ndenumerate(M):
        if value!=0 and x!=y:
            tuples.append(((x+min_value,y+min_value),value))

    C = sorted(tuples, key=lambda x: x[1])
	# #print "done"
    # t = np.count_nonzero(adj_matrix)
	# #print(t)
    # C = C[-t:]

	#print(C)
	#print 'C done'

    ln = len(C)
    ln = ln-1

    nodes =dict()
    root_nodes = set()
    tree = Tree()

    # for index in tqdm(range(ln, -1, -1)):
    for index in range(ln, -1, -1):
        vertices, value = C[index]
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

	#print tree.root.vertices, len(tree.root.vertices)
    root_nodes = list(filter( lambda entry: entry.parent==None, list(root_nodes)))

    return tree, root_nodes, nodes, G


def extract_communities(tree, root_nodes, nodes, G, min_threshold):
    tree.communities = []
    # for temp_roots in tqdm(root_nodes):
    for temp_roots in root_nodes:

        tree.root = temp_roots
		#Counting number of vertices and Edges
        tree.count_vertices_and_edges(G.edges(),nodes)
		#Summing up number of edges of children to parent
        tree.count_vertices_and_edges_wrap(tree.root)
		#Computing density of Tree Nodes
        tree.compute_density(tree.root)
		#Filtering Nodes as Per Density Threshold
        tree.extract_sub_graph(tree.root, min_threshold)

    return tree.communities


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