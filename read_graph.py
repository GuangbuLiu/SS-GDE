import numpy as np
import os, sys
import scipy.sparse as sp
import networkx as nx
import pickle as pkl
from config import Graph
from dgc_funs import *
import torch
import time
import pdb
eps = np.finfo(np.float32).eps
#Read graphs in the MUTAG data collection
def read_graph(fname):
    EDGE_FILE = "edgelist.txt"
    node_labels = list()
    edge_labels = list()
    graph_label = None
    read_vert = False
    read_edge = False
    read_al = False #read adjacency list
    read_graph_class = False
    
    with open(fname, "rb") as graph:
        with open(EDGE_FILE, "wb") as edgelist_file:
            
            lines = graph.readlines()
            for line in lines:
                if line.startswith("#v"):
                    read_vert = True
                    continue
                if line.startswith("#e"):
                    read_vert = False
                    read_edge = True
                    continue
                if line.startswith("#a"):
                    read_vert = False
                    read_al = True
                    node_count = 1
                    continue
                if line.startswith("#c"):
                    read_edge = False
                    read_al = False
                    read_graph_class = True
                    continue

                if read_vert:
                    node_labels.append(int(line))
                elif read_edge: #format: v1,v2,label
                    edge_info = line.split(",")
                    edgelist_file.write(edge_info[0] + " " + edge_info[1] + " " + edge_info[2] + "\n")
                    edge_labels.append(int(edge_info[2]))
                elif read_al: #format: for node i (i lines after the adjacency list start), list of all nodes it's adj to
                    adj_list = line.split(",")
                    for neighbor in adj_list:
                        edgelist_file.write(str(node_count) + " " + neighbor + "\n")
                    node_count += 1
                elif read_graph_class:
                    graph_label = int(line)

    nx_graph = nx.read_edgelist(EDGE_FILE, nodetype = int, data=(('label',int),))
    adj = nx.adjacency_matrix(nx_graph)#.todense()
    graph = Graph(adj, node_labels = node_labels, edge_labels = nx.adjacency_matrix(nx_graph, weight = "label"), graph_label = graph_label)
    os.system("rm " + EDGE_FILE) #clean up
    return graph

#Read fron https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets
#TODO DECIDE should we copy graph.node_labels to graph.node_features?
def read_combined(dataset_name, remove_small = None, read_edge_labels = False):
    graphs = list()
    path = os.path.join("../data/benchmarks/" + dataset_name + "/" + dataset_name)
    # path = os.path.join(os.path.dirname(__file__), "benchmarks/" + dataset_name + "/" + dataset_name)
    adj_fname = path + "_A.txt"
    graph_labels = read_list(path + "_graph_labels.txt")
    graph_indicators = read_list(path + "_graph_indicator.txt")
    print(path + "_node_labels.txt")
    if os.path.exists(path + "_node_labels.txt"):
        print("found node labels at path %s" % (path + "_node_labels.txt"))
        node_labels = read_list(path + "_node_labels.txt")
    else:
        node_labels = None

    if os.path.exists(path + "_edge_labels.txt"):
        edge_labels = read_list(path + "_edge_labels.txt")
    else:
        print("no edge labels to read")
        edge_labels = None
    
    dim_starts = get_graph_starts(graph_indicators)

    if dataset_name in ["MUTAG", "PTC_MR"]:
        delimiter = ", "
    else:
        delimiter = ","
    
    nx_graph = nx.read_edgelist(path + "_A.txt", delimiter=delimiter, nodetype = int)

    combined_adj = nx.adjacency_matrix(nx_graph, nodelist = range(1, max(nx_graph.nodes) + 1))#.todense()
    combined_adj = sp.csc_matrix(combined_adj)
    #print "Combined shape: ", combined_adj.shape
    
    if read_edge_labels and edge_labels is not None: #we don't use edge labels
        indices = read_list(path + "_A.txt", dtype = "string")
        rows = [int(x.split(",")[0]) - 1 for x in indices]
        cols = [int(x.split(",")[1]) - 1 for x in indices]
        sp_mat = sp.csr_matrix((edge_labels, (rows, cols)), shape = combined_adj.shape)
        combined_edge_labels = sp_mat
    embs = []
    for i in range(len(dim_starts) - 1):
        '''
        indiv_nx = nx_graph.subgraph(1 + np.arange(dim_starts[i], dim_starts[i + 1])) #since node IDs start from 1
        indiv_adj= nx.adjacency_matrix(indiv_nx)
        '''
        indiv_adj = combined_adj[:, dim_starts[i]:dim_starts[i + 1]] #slice columns, which are efficient
        indiv_adj = indiv_adj.tocsr() #for effective row slicing
        indiv_adj = indiv_adj[dim_starts[i]:dim_starts[i + 1]]
        #indiv_adj = combined_adj[dim_starts[i]:dim_starts[i + 1], dim_starts[i]:dim_starts[i + 1]]
        indiv_adj = sp.csr_matrix(indiv_adj)
  
        if read_edge_labels and edge_labels is not None:
            indiv_edge_labels = combined_edge_labels[dim_starts[i]:dim_starts[i + 1], dim_starts[i]:dim_starts[i + 1]]
        else:
            indiv_edge_labels = None

        if node_labels is not None:
            indiv_node_labels = node_labels[dim_starts[i]:dim_starts[i+1]]
        else:
            indiv_node_labels = None
            
        embs.append(convert_data(indiv_adj.toarray()))
        graph = Graph(adj = indiv_adj, node_labels = indiv_node_labels, edge_labels = indiv_edge_labels, graph_label = graph_labels[i])
        if remove_small is not None: #only keep graphs if they have number of vertices above a given threshold
            if graph.N > remove_small:
                graphs.append(graph)
        else:
            graphs.append(graph)

    '''Test'''
    '''
    for i in range(len(graphs)):
        if graphs[i].node_labels is not None:
            assert graphs[i].G_adj.shape[0] == len(graphs[i].node_labels)
    '''
    return graphs,embs

#Read text file with column headers
#Columns: node ID, label
def read_labels(labels_file):
    labels_dict = {}
    with open(labels_file, "rb") as lf:
        node_labels = lf.readlines()[1:] #first line is column header
    for node in node_labels:
        n = node.split()
        labels_dict[int(n[0])] = int(n[1])
    return labels_dict

#Read in a list whose entries are one line
def read_list(fname, dtype = "float"):
    with open(fname, "rb") as f:
        lines = np.asarray(f.readlines(), dtype = dtype)
        return lines

#Given a list of indicators of nodes, give an index of where the graph should start
#Assume in sorted order
def get_graph_starts(graph_indicators):

    graph_starts = [0]
    for i in range(1, len(graph_indicators)):
        if graph_indicators[i] > graph_indicators[i - 1]:
            graph_starts.append(i)
    graph_starts.append(len(graph_indicators))
    return graph_starts

def get_node_max(embs,max_up):
    
    for i in range(len(embs)):
        if i==0:
            maxnnode = embs[i].shape[0]
        elif maxnnode < embs[i].shape[0]:
            maxnnode = embs[i].shape[0] 
    if maxnnode>=max_up:
        maxnnode = max_up
    return maxnnode

def get_node_unify(embs,adj,dim,max_up):
    
    for i in range(len(embs)):
        
        adj_i = adj[i]
        adj_diag = np.diag((adj_i.sum(-1)+eps)**(-0.5))
        adj[i] = np.matmul(np.matmul(adj_diag,adj_i),adj_diag)
        embs_row = embs[i].shape[0]
        
        if embs_row>max_up:
            # t = 1000 * time.time()
            # np.random.seed(int(t) % 2**32)
            index = np.random.choice(embs_row,max_up)
            index = np.sort(index)
            embs[i] = embs[i][index]
            adj[i] = adj[i][index,:]
            adj[i] = adj[i][:,index]
            adj[i] = adj[i].astype(np.float32)

        else:
           
            embs_zeros_add = np.zeros((max_up - embs_row,dim), dtype=np.float32)
            embs[i] = np.concatenate((embs[i],embs_zeros_add),axis=0)
            adj_zeros_add_row = np.zeros((max_up - embs_row,embs_row))
            adj[i] = np.concatenate((adj[i],adj_zeros_add_row),axis=0)
            adj_zeros_add_column = np.zeros(((max_up,max_up-embs_row)))
            adj[i] = np.concatenate((adj[i],adj_zeros_add_column),axis=1)
            adj[i] = adj[i].astype(np.float32)
    
    adj = [_a.astype(np.float32) for _a in adj]
    new_adj = torch.tensor(adj,dtype=torch.float32)
    # for j in range(len(adj)):
    #     print(j)
    #     if j ==0:
    #         new_adj = torch.tensor(adj[j])
    #     else:
    #         new_adj = torch.cat((new_adj,torch.tensor(adj[j])),0)
    # new_adj = new_adj.view(len(adj),-1,new_adj.shape[-1])
    # pdb.set_trace()
    return embs,new_adj

def get_adj_unify(adj,maxnnode,max_up):
    
    for i in range(len(adj)):
        adj_i = adj[i]
        adj_diag = np.diag((adj_i.sum(-1)+eps)**(-0.5))
        adj[i] = np.matmul(np.matmul(adj_diag,adj_i),adj_diag)
   
    if maxnnode>=max_up:
        maxnnode = max_up
    for i in range(len(adj)):
        adj_row = adj[i].shape[0]
        if adj_row>=maxnnode:
            adj[i] = adj[i][0:maxnnode,0:maxnnode]
        else:
            adj_zeros_add_row = np.zeros((maxnnode - adj_row,adj_row))
            adj[i] = np.concatenate((adj[i],adj_zeros_add_row),axis=0)
            adj_zeros_add_column = np.zeros(((maxnnode,maxnnode-adj_row)))
            adj[i] = np.concatenate((adj[i],adj_zeros_add_column),axis=1)

    adj = [_a.astype(np.float32) for _a in adj]
    
    for j in range(len(adj)):
        if j ==0:
            new_adj = torch.tensor(adj[j])
        else:
            new_adj = torch.cat((new_adj,torch.tensor(adj[j])),0)
    new_adj = new_adj.view(len(adj),-1,new_adj.shape[-1])
    
    return new_adj


if __name__ == "__main__":
    read_graph("data/graph_similarity/enzymes/enzymes_1.graph")

