import math
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
from scipy import sparse
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
import pdb


class Graph():
    # Undirected, unweighted
    def __init__(self, adj, node_labels=None, edge_labels=None, graph_label=None, node_attributes=None,
                 node_features=None, graph_id=None):
        self.graph_id = graph_id
        self.adj = adj  # adjacency matrix
        self.N = self.adj.shape[0]  # number of nodes

        if node_features is None:
            self.node_features = {}
        else:
            self.node_features = node_features

        self.max_features = {}
        for feature in self.node_features:
            self.max_features[feature] = np.max(self.node_features[feature])

        self.node_labels = node_labels
        self.edge_labels = edge_labels
        self.graph_label = graph_label
        self.node_attributes = node_attributes  # N x A matrix, where N is # of nodes, and A is # of attributes

    # '''
    def set_node_features(self, node_features):
        self.node_features = node_features
        for feature in self.node_features:  # TODO hacky to handle this case separately
            self.max_features[feature] = np.max(self.node_features[feature])

    def compute_node_features(self, features_to_compute):
        nx_graph = to_nx(self.adj)
        new_node_features = self.node_features
        if "degree" in features_to_compute:
            total_degrees = nx_graph.degree(nx_graph.nodes())
            new_node_features["degree"] = total_degrees

        self.set_node_features(new_node_features)

    def normalize_node_features(self):
        normalized_features_dict = dict()
        for feature in self.node_features:
            normalized_features = self.node_features[feature]
            if np.min(normalized_features) < 0:  # shift so no negative values
                normalized_features += abs(np.min(normalized_features))
            # scale so no feature values less than 1 (for logarithmic binning)
            if np.min(normalized_features) < 1:
                normalized_features /= np.min(normalized_features[normalized_features != 0])
                if np.max(normalized_features) == 1:  # e.g. binary features
                    normalized_features += 1
                normalized_features[
                    normalized_features == 0] = 1  # set 0 values to 1--bin them in first bucket (smallest values)
            normalized_features_dict[feature] = normalized_features
        self.set_node_features(normalized_features_dict)


# '''

def to_nx(adj):
    if sparse.issparse(adj):
        return nx.from_scipy_sparse_matrix(adj)
    else:
        return nx.from_numpy_matrix(adj)

class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, bias=False):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        # self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.weight = nn.init.kaiming_normal_(torch.Tensor(in_features, out_features),mode='fan_out', nonlinearity='relu')
        self.weight = Parameter(self.weight)
        # self.weight = Parameter(torch.Tensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        # self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        # self.weight = nn.init.kaiming_normal_(p,mode='fan_out', nonlinearity='relu')
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
      
        support = torch.matmul(input, self.weight)
        output = torch.matmul(adj, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'
