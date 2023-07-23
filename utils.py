import math
import torch
import numpy as np
import mpu.ml as mml
import scipy.sparse
import torch.utils.data.dataset as Dataset
eps = np.finfo(np.float32).eps

def get_node_max(embs, max_up):
    for i in range(len(embs)):
        if i == 0:
            maxnnode = embs[i].shape[0]
        elif maxnnode < embs[i].shape[0]:
            maxnnode = embs[i].shape[0]
    if maxnnode >= max_up:
        maxnnode = max_up
    return maxnnode


def get_node_unify(embs, adj, dim, max_up):
    for i in range(len(embs)):

        adj_i = adj[i]
        adj_diag = np.diag((adj_i.sum(-1) + eps) ** (-0.5))
        adj[i] = np.matmul(np.matmul(adj_diag, adj_i), adj_diag)
        embs_row = embs[i].shape[0]

        if embs_row > max_up:
            # t = 1000 * time.time()
            # np.random.seed(int(t) % 2**32)
            index = np.random.choice(embs_row, max_up)
            index = np.sort(index)
            embs[i] = embs[i][index]
            adj[i] = adj[i][index, :]
            adj[i] = adj[i][:, index]
            adj[i] = adj[i].astype(np.float32)

        else:

            embs_zeros_add = np.zeros((max_up - embs_row, dim), dtype=np.float32)
            embs[i] = np.concatenate((embs[i], embs_zeros_add), axis=0)
            adj_zeros_add_row = np.zeros((max_up - embs_row, embs_row))
            adj[i] = np.concatenate((adj[i], adj_zeros_add_row), axis=0)
            adj_zeros_add_column = np.zeros(((max_up, max_up - embs_row)))
            adj[i] = np.concatenate((adj[i], adj_zeros_add_column), axis=1)
            adj[i] = adj[i].astype(np.float32)

    adj = [_a.astype(np.float32) for _a in adj]
    new_adj = torch.tensor(adj, dtype=torch.float32)

    return embs, new_adj


class subDataset(Dataset.Dataset):

    def __init__(self, idx_train):
        super(subDataset, self).__init__()
        self.idx_mode = idx_train

    def __len__(self):
        return len(self.idx_mode)  # must to return the len

    def __getitem__(self, index):
        idx_mode = self.idx_mode[index]

        return idx_mode  # dimen


class Calculate_acc(torch.nn.Module):

    def __init__(self):
        super(Calculate_acc, self).__init__()
        self.scores = torch.tensor([], dtype=torch.int64).cuda()
        self.gt = torch.tensor([], dtype=torch.int64).cuda()

    def add(self, final_scores, graph_labels, new_labels_unique):
        # new_labels_unique = torch.unique(graph_labels)
        max_score, index = torch.max(final_scores, dim=1)
        pred_labels = new_labels_unique[index].cuda()
        self.scores = torch.cat((self.scores, pred_labels), dim=0)
        self.gt = torch.cat((self.gt, graph_labels), dim=0)

    def calculate_acc(self):
        if len(self.gt) != len(self.scores):
            pdb.set_trace()

        correct = self.scores.eq(self.gt).long()

        correct = correct.sum().float()
        return correct / (len(self.gt) + eps)

def convert_data(data):
    ## data.keys(): ['graph', 'labels']
    ## -- output >>
    ##              As: list,  adjacent matrix (sparse)
    ##              labels: vector
    ##              nodelables: list of vectors
    ##              nnodes: vector

    constant = 10
    labels = np.asarray(data['labels'], dtype=np.int32) + constant
    labels = np.ravel(labels)
    labels[labels == -1] = 0  # only 0, 1, 2
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

        m = 2 * m - missing_node

        A = np.zeros((m, m), dtype=np.float32)
        nodeL = np.zeros(m, dtype=np.int32)

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


def to_onehot(gt, num_cls=59):
    gt = gt.reshape(-1)
    gt = mml.indices2one_hot(gt, nb_classes=num_cls)
    gt = np.asarray(gt)

    return gt

def queue_idx_each_class(labels_num, graph_labels, num_keys):
    # queue_idx_class = np.array([])
    for i in range(len(labels_num)):

        if type(graph_labels) == torch.Tensor:
            graph_labels = np.array(graph_labels.cpu())
        class_i = np.where(graph_labels == labels_num[i])[0]
        index_class_i = np.random.choice(class_i, num_keys).reshape(1, -1)
        if i == 0:
            queue_idx_class = index_class_i
        else:
            queue_idx_class = np.concatenate((queue_idx_class, index_class_i), axis=1)
    queue_idx_class = np.squeeze(queue_idx_class)
    return queue_idx_class

def lr_update_rule(epoch):
    lamda = math.pow(0.8, int(epoch / 40))
    return lamda