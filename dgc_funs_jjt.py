'''
dgc_funs.py
'''
import numpy as np
import scipy.sparse
import types
import mpu.ml as mml
import pdb
'''
convert_data(data)
'''

def convert_data(data):
    ## data.keys(): ['graph', 'labels']
    ## -- output >>
    ##              As: list,  adjacent matrix (sparse)
    ##              labels: vector
    ##              nodelables: list of vectors
    ##              nnodes: vector

    constant = 10
    labels = np.asarray(data['labels'],dtype=np.int32) + constant
    labels = np.ravel(labels)
    labels[labels==-1] = 0 # only 0, 1, 2 
    n = len(labels)
    print('len(labels):{}'.format(len(labels)))
    As = []
    nodelabels = []
    nnodes = np.zeros(n, dtype=np.int32)
    nedges = np.zeros(n, dtype=np.int32)
    
    for ii in range(n):
        x = data['graph'][ii]
        m = len(x)
        # calculate missing node
        missing_node = 0
        for jj in range(m):
            # if x.has_key(jj):
            if jj in x:
                missing_node += 1

        m = 2*m - missing_node

        A = np.zeros((m,m),dtype=np.float32)
        nodeL = np.zeros(m,dtype=np.int32)

        for jj in range(m):
            if jj in x:
            # if x.has_key(jj):
                A[jj, x[jj]['neighbors']] = 1
                l = x[jj]['label']
                if type(l) is tuple and len(l) == 1:
                # if type(l) is types.TupleType and len(l) == 1:
                    nodeL[jj] = l[0] + constant
                else:
                    nodeL[jj] = len(x[jj]['neighbors']) + constant
                    # assert(len(l)==1)
                    # nodeL[jj] = l[0] + constant

        A = scipy.sparse.coo_matrix(A)
        As.append(A)
        nodelabels.append(nodeL)
        nnodes[ii] = m
        nedges[ii] = A.nnz

    # nodelabels must be ordered: 1, 2, 3, ...
    node_unique = np.unique(np.concatenate(nodelabels))
    for index, value in enumerate(node_unique):
        for i in range(len(nodelabels)):
            for j in range(len(nodelabels[i])):

                if nodelabels[i][j] == value:
                    nodelabels[i][j] = index + 1

    # labels must be ordered: 0, 1, 2, ... and shape is (n,)
    label_unique = np.unique(labels)
    for index, value in enumerate(label_unique):
        for i in range(len(labels)):
            if labels[i] == value:
                labels[i] = index



    return As, labels, nodelabels, nnodes, nedges
    

def one_hot_embs(nodelabels,maxnode_num,class_max_nnode):
    
    graph_num = len(nodelabels)

    # for i in range(graph_num):
    #     graph_embs = np.zeros((class_max_nnode,maxnode_num),dtype=float)
    #     graph_i = nodelabels[i]
    #     embs_row = len(graph_i)
    #     embs_column = len(np.unique(graph_i))

    return graph_num

def to_onehot(gt, num_cls=59):
   
    # num_cls = len(np.unique(gt))
    gt_shape = gt.shape
    gt = gt.reshape(-1)
    gt = mml.indices2one_hot(gt, nb_classes=num_cls)
    gt = np.asarray(gt)
    # gt = np.asarray(gt).transpose()
    # gt = gt.reshape(-1, *gt_shape)    
    return gt

            
