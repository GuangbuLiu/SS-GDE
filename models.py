import time

import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
import torch
import pdb


class GCN(nn.Module):
    def __init__(self, nfeat, first_hidden, second_hidden, third_hidden, dropout):
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, first_hidden)
        self.gc2 = GraphConvolution(first_hidden, second_hidden)
        self.gc3 = GraphConvolution(second_hidden, third_hidden)

        self.relu = nn.ReLU(inplace = True)
        self.dp = nn.Dropout(p=dropout)

    def forward(self, x, adj):

        x = torch.unsqueeze(x,dim=1)
        adj = torch.unsqueeze(adj,dim=1)

        x = self.relu(self.gc1(x, adj))
        x = self.relu(self.gc2(x, adj))
        x = self.gc3(x, adj)

        x = torch.squeeze(x)
        return x


class SS_GDE(nn.Module):
    def __init__(self, args, dict, dict_adj, dict_nnode, dict_label):
        super(SS_GDE, self).__init__()

        self.args = args
        self.expect_sampling_prob = args.expect_prob

        # sinkhorn coefficient
        lamda = torch.tensor([0.01, 0.1, 0.5, 1, 3, 5, 10, 20])
        self.lamda = lamda.reshape(lamda.size(0), 1, 1, 1, 1).cuda()

        self.encoder = GCN(nfeat=args.dimensionality,
                        first_hidden=args.first_hidden,
                        second_hidden=args.second_hidden,
                        third_hidden=args.third_hidden,
                        dropout=args.dropout)

        self.dict_encoder = GCN(nfeat=args.dimensionality,
                        first_hidden=args.first_hidden,
                        second_hidden=args.second_hidden,
                        third_hidden=args.third_hidden,
                        dropout=args.dropout)

        self.relu = nn.ReLU(inplace=True)

        self.dist_para = torch.randn([args.max_nnode * args.dict_max_nnode, 32])
        self.dist_para = torch.nn.Parameter(self.dist_para, requires_grad=True)

        # classifier
        self.fc1 = nn.Linear(32 * args.classes * args.num_keys, 64, bias=True)
        self.fc1.weight = torch.nn.Parameter(self.fc1.weight, requires_grad=True)
        self.fc1.bias = torch.nn.Parameter(self.fc1.bias, requires_grad=True)

        self.fc2 = nn.Linear(64, args.classes, bias=True)
        self.fc2.weight = torch.nn.Parameter(self.fc2.weight, requires_grad=True)
        self.fc2.bias = torch.nn.Parameter(self.fc2.bias, requires_grad=True)

        # bernoulli sampling factor learning
        self.mlp = nn.Linear(args.max_nnode, 1)
        self.mlp.weight = torch.nn.Parameter(self.mlp.weight, requires_grad=True)
        self.mlp.bias = torch.nn.Parameter(self.mlp.bias, requires_grad=True)

        self.mlp2 = nn.Linear(args.batch_size, 1)
        self.mlp2.weight = torch.nn.Parameter(self.mlp2.weight, requires_grad=True)
        self.mlp2.bias = torch.nn.Parameter(self.mlp2.bias, requires_grad=True)

        # attention weight
        self.weight_lamda = torch.randn(args.classes * args.num_keys * 32, 1)
        self.weight_lamda = torch.nn.Parameter(self.weight_lamda, requires_grad=True)

        # dictionary keys
        self.dict = dict
        self.dict = torch.nn.Parameter(self.dict, requires_grad=True)
        self.dict_adj = dict_adj.cuda()
        self.dict_nnode = dict_nnode
        self.dict_label = dict_label.cuda()

    def dist(self, x1, x2):

        x1 = torch.unsqueeze(x1, dim=2)
        x1 = torch.unsqueeze(x1, dim=1)
        x2 = torch.unsqueeze(x2, dim=1)
        x2 = torch.unsqueeze(x2, dim=0)

        dist = x1 - x2
        dist = torch.sum(dist ** 2, dim=-1)
        torch.cuda.empty_cache()
        return dist

    def cosine_sim(self,x, y):
        x = x.unsqueeze(dim=1)
        y = y.unsqueeze(dim=0)
        x = x.div(torch.norm(x, p=2, dim=2, keepdim=True) + 1e-12)
        y = y.div(torch.norm(y, p=2, dim=2, keepdim=True) + 1e-12)
        cos_dis = torch.matmul(x, y.transpose(2,3))
        return cos_dis

    def kl_divergence(self, prob):

        expect_prob = torch.ones_like(prob) * self.expect_sampling_prob

        kl_div = prob * torch.log(prob / expect_prob) + (1 - prob) * torch.log((1 - prob) / (1 - expect_prob))

        kl_loss = torch.sum(kl_div)

        return kl_loss

    def VGDA(self, x, dict):

        # Since the last dataloader is less than batch_size
        x_zero = torch.zeros(self.args.batch_size - x.size(0), x.size(1), x.size(2)).cuda()
        x_cat = torch.cat((x, x_zero), dim=0)

        sims = self.cosine_sim(dict, x_cat)
        sims = torch.transpose(sims, 1, 2)

        prob = self.mlp(sims).squeeze(-1)
        prob = self.mlp2(prob).squeeze(-1)

        attn = torch.sigmoid(prob)
        bernoulli_attn = torch.bernoulli(attn).unsqueeze(-1)
        dict = bernoulli_attn * dict

        kl_loss = self.kl_divergence(attn)

        return dict, kl_loss

    def sinkhorn(self, w1, w2, M, graph_nnode_i, graph_nnode_j, lamda, k, eps = 1e-6):
        """Sinkhorn algorithm with fixed number of iteration (autograd)
        """

        K = torch.exp(-M * lamda)

        mask = torch.zeros(K.shape[1], K.shape[2], K.shape[3], K.shape[4]).cuda()
        for i in range(K.shape[1]):
            mask[i, :, :graph_nnode_i[i], :] = 1

        for j in range(K.shape[2]):
            mask[: ,j , :, graph_nnode_j[j]:] = 0
        mask = mask.unsqueeze(0).repeat(K.shape[0], 1, 1, 1, 1)

        K_mask = K * mask
        K_mask = K_mask.reshape(K_mask.shape[0] * K_mask.shape[1], K_mask.shape[2], K_mask.shape[3], K_mask.shape[4])

        ui = torch.ones((K_mask.shape[0], K_mask.shape[1], K_mask.shape[2],))
        vi = torch.ones((K_mask.shape[0], K_mask.shape[1], K_mask.shape[3],))
        ui = torch.unsqueeze(ui, dim=-1).cuda()
        vi = torch.unsqueeze(vi, dim=-1).cuda()
        w1 = w1.repeat(lamda.shape[0], 1)
        w1 = torch.unsqueeze(w1, dim=0).cuda()
        w1 = torch.unsqueeze(w1, dim=-1).cuda()
        w2 = torch.unsqueeze(w2, dim=0).cuda()
        w2 = torch.unsqueeze(w2, dim=-1).cuda()

        for i in range(k):
            vi = torch.div(w2, (torch.matmul(K_mask.transpose(2, 3), ui) + eps))
            ui = torch.div(w1, (torch.matmul(K_mask, vi) + eps).transpose(0, 1))
            ui = ui.transpose(0, 1)

        G = ui * K_mask * vi.transpose(2, 3)

        # normalize
        G_max, _ = torch.max(G, dim=-1)
        G_max = torch.unsqueeze(G_max, dim=-1)
        G_min, _ = torch.min(G, dim=-1)
        G_min = torch.unsqueeze(G_min, dim=-1)
        G = torch.div(G - G_min, G_max - G_min + eps)

        torch.cuda.empty_cache()
        return G

    def MS_WE(self, X, dict, nnode, k=10):
        # pdb.set_trace()
        wi = torch.FloatTensor(torch.ones([X.shape[0], X.shape[1]])) / nnode.float().view(-1, 1)
        for i in range(len(nnode)):
            wi[i, nnode[i]:] = 0

        wj = torch.FloatTensor(torch.ones([dict.shape[0], dict.shape[1]])) / self.dict_nnode.float().view(-1, 1)
        for j in range(len(self.dict_nnode)):
            wj[j, self.dict_nnode[j]:] = 0

        M = self.dist(X, dict)
        G = self.sinkhorn(wi, wj, M, nnode, self.dict_nnode, self.lamda, k)
        M = M.repeat(self.lamda.shape[0], 1, 1, 1)
        graph_dist = G * M.cuda()

        # attention mechanism
        # graph_dist: (num_lamda * batchsize, num_keys, N1, N2)
        graph_dist = graph_dist.view(graph_dist.shape[0], graph_dist.shape[1], -1)
        # pdb.set_trace()
        graph_dist = torch.matmul(graph_dist, self.dist_para)
        graph_dist_all_order = graph_dist.view(graph_dist.shape[0], -1)
        graph_dist_all_order = graph_dist.reshape(self.lamda.shape[0], X.shape[0], graph_dist_all_order.shape[1])

        # attention coefficient
        coeff = torch.matmul(graph_dist_all_order, self.weight_lamda)  # weight_lamda:640*1
        coeff = F.softmax(coeff, dim=0)
        graph_dist_all_order = graph_dist_all_order * coeff
        graph_dist_all_order = torch.sum(graph_dist_all_order, 0)
        return graph_dist_all_order

    def forward(self, x, adj, num_node):

        # encode input graph and graph dictionary
        x = self.encoder(x, adj)
        dict = self.dict_encoder(self.dict, self.dict_adj)

        # Get the personalized dictionary by VGDA module
        dict, kl_loss = self.VGDA(x, dict)

        # MS-WE module
        embed = self.MS_WE(x, dict, num_node)

        # classify
        out = self.fc1(embed)
        out = self.fc2(out)

        # dictionary loss
        dict_loss = self.dict_loss(dict)

        return out, dict_loss, kl_loss


    def split_classes(self, X, y):

        lstclass = torch.unique(y, sorted=True)
        return [X[y == i].float() for i in lstclass], [y == i for i in lstclass], lstclass

    def dict_loss(self, dict, reg=1, k=10, eps = 1e-6):

        xc, graph_class_index, lstclass = self.split_classes(dict.cpu(), self.dict_label)
        queue_nnode_each = self.dict_nnode.view(len(xc), -1)
        loss_b = 0
        loss_w = 0
        graph_dist_list = []

        for i, xi in enumerate(xc[:]):
            xc_nnode_i = self.dict_nnode[graph_class_index[i]]
            queue_nnode_i = queue_nnode_each[i]
            wi = torch.FloatTensor(torch.ones([xi.shape[0], xi.shape[1]])) / (xc_nnode_i.view(-1, 1).float() + eps)
            for ii in range(len(xc_nnode_i)):
                wi[ii, xc_nnode_i[ii]:] = 0

            for j, xj in enumerate(xc[:]):
                xc_nnode_j = self.dict_nnode[graph_class_index[j]]
                queue_nnode_j = queue_nnode_each[j]
                wj = torch.FloatTensor(torch.ones([xj.shape[0], xj.shape[1]])) / (xc_nnode_j.view(-1, 1).float() + eps)
                for jj in range(len(xc_nnode_j)):
                    wj[jj, xc_nnode_j[jj]:] = 0
                M = self.dist(xi.cuda(), xj.cuda())

                G = self.sinkhorn_center(wi, wj, M, xc_nnode_i, xc_nnode_j, reg, k)
                graph_dist = G * M.cuda()

                graph_dist_column = torch.sum(graph_dist, dim=-1)
                queue_nnode_i = torch.unsqueeze(queue_nnode_i, dim=-1)
                queue_nnode_i = torch.unsqueeze(queue_nnode_i, dim=0).cuda()
                graph_dist_column = torch.div(graph_dist_column, queue_nnode_i.float() + eps)
                graph_dist_row = torch.sum(graph_dist_column, dim=-1)
                queue_nnode_j = torch.unsqueeze(queue_nnode_j, dim=-1).cuda()
                graph_dist_all = torch.div(graph_dist_row, queue_nnode_j.float() + eps)

                queue_nnode_i = torch.squeeze(queue_nnode_i)
                queue_nnode_j = torch.squeeze(queue_nnode_j)
                graph_dist_list.append(torch.squeeze(graph_dist_all))

                if j == i:
                    loss_w = loss_w + torch.sum(graph_dist_all)  # G*M
                elif i < j:
                    loss_b = loss_b + torch.sum(graph_dist_all)

        return loss_w / (loss_b + eps)

    def sinkhorn_center(self, w1, w2, M, graph_nnode_i, graph_nnode_j, reg, k, eps = 1e-6):
        """Sinkhorn algorithm with fixed number of iteration (autograd)
        """

        K = torch.exp(-M * reg)
        mask = torch.zeros(M.shape).cuda()

        for i in range(M.shape[0]):
            for j in range(M.shape[1]):
                mask[i][j][0:graph_nnode_i[i], 0:graph_nnode_j[j]] = 1
        K_mask = K * mask

        ui = torch.ones((M.shape[0], M.shape[1], M.shape[2],))
        vi = torch.ones((M.shape[0], M.shape[1], M.shape[3],))
        ui = torch.unsqueeze(ui, dim=-1).cuda()
        vi = torch.unsqueeze(vi, dim=-1).cuda()
        w1 = torch.unsqueeze(w1, dim=0).cuda()
        w1 = torch.unsqueeze(w1, dim=-1).cuda()
        w2 = torch.unsqueeze(w2, dim=0).cuda()
        w2 = torch.unsqueeze(w2, dim=-1).cuda()

        for i in range(k):
            vi = torch.div(w2, (torch.matmul(K_mask.transpose(2, 3), ui) + eps))
            ui = torch.div(w1, (torch.matmul(K_mask, vi) + eps).transpose(0, 1))
            ui = ui.transpose(0, 1)

        G = ui * K_mask * vi.transpose(2, 3)

        G_max, _ = torch.max(G, dim=-1)
        G_max = torch.unsqueeze(G_max, dim=-1)
        G_min, _ = torch.min(G, dim=-1)
        G_min = torch.unsqueeze(G_min, dim=-1)
        G = torch.div(G - G_min, G_max - G_min + eps)
        return G