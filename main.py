import pdb
import torch
import torch.optim as optim
import torch.utils.data.dataloader as DataLoader

from utils import queue_idx_each_class, lr_update_rule
from embed import *
from read_graph import *
from dgc_funs_jjt import *
from utils_single import *
from models import SS_GDE


# Training settings
parser = argparse.ArgumentParser()
parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
parser.add_argument('--dataset', nargs='?', default='proteins', help='dataset (mutag, nci, ptc, proteins, imdb)')
parser.add_argument('--seed', type=int, default=777, help='Random seed.')
parser.add_argument('--epochs', type=int, default=500, help='Number of epochs to train.')
parser.add_argument('--nfold', type=int, default=10, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Number of samples per dataloader')
parser.add_argument('--num_keys', type=int, default=14, help='Number of graph keys')
parser.add_argument('--max_nnode', type=int,default=128, help='Max number of input graph nodes')
parser.add_argument('--dict_max_nnode', type=int, default=64, help='Max number of dictionary nodes')
parser.add_argument('--kl_loss_ratio', type=float, default=0.001, help='Ratio of KL-divergence loss')
parser.add_argument('--m', type=float, default=0.995, help='Momentum coefficient')
parser.add_argument('--lr', type=float, default=0.001, help='Initial learning rate.')  #0.01
parser.add_argument('--lr_step', type=int, default=40, help='learning rate update steps.')
parser.add_argument('--weight_decay', type=float, default=10e-4, help='Weight decay (L2 loss on parameters).')
parser.add_argument('--first_hidden', type=int, default=256, help='Number of hidden units.')
parser.add_argument('--second_hidden', type=int, default=128, help='Number of hidden units.')
parser.add_argument('--third_hidden', type=int, default=32, help='Number of hidden units.')
parser.add_argument('--dropout', type=float, default=0, help='Dropout rate (1 - keep probability).')
parser.add_argument('--test_fre', type=int, default=2, help='test frequency per epoch.')
parser.add_argument('--expect_prob', type=float, default=0.5, help='Expect probability in KL-divergence')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
print(args)

# Load data
fpath = os.path.join('../data/benchmarks/', '{}.graph'.format(args.dataset))
with open(fpath, 'rb') as fp:
    data = pickle.load(fp, encoding='iso-8859-1')

As, labels, nodelabels, nnodes, nedges = convert_data(data)

# class of node labels
maxnode_num = len(np.unique(np.concatenate(nodelabels)))
args.dimensionality = maxnode_num

graph_labels = np.array(labels, dtype=np.int64)
embs = [to_onehot(k - 1, args.dimensionality) for k in nodelabels]

labels_unique = np.unique(graph_labels)
labels_num = np.arange(len(labels_unique))
graphs = labels

args.classes = len(labels_unique)
args.dimensionality = maxnode_num

graph_adj = [graph_i.toarray() for graph_i in As]
for i in range(len(graph_adj)):
    for j in range((graph_adj[i].shape[0])):
        for k in range((graph_adj[i].shape[1])):
            if nodelabels[i][j] == nodelabels[i][k]:
                graph_adj[i][j][k] = 1

embs = [emb.astype(np.float32) for emb in embs]
graph_nnode = torch.tensor([embs[i].shape[0] for i in range(len(embs))])

graph_nnode = torch.where(graph_nnode >= torch.tensor(args.max_nnode), torch.tensor(args.max_nnode),
                          graph_nnode)
maxnnode = get_node_max(embs, args.max_nnode)

args.max_nnode = maxnnode

embs_unify, adj_unify = get_node_unify(embs, graph_adj, args.dimensionality, maxnnode)
embs_unify = torch.tensor(embs_unify)


def test(idx_test_i):
    model.eval()
    dataset_test = subDataset(idx_test_i)
    dataloader_test = DataLoader.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False,
                                            num_workers=4, drop_last=False)
    # correct = 0
    test_loss_list = []
    Calculate_acc_test = Calculate_acc()
    for img_num_i, data in enumerate(dataloader_test):
        idx_test_i = data.cuda()

        x = embs_unify[idx_test_i].cuda()
        adj = adj_unify[idx_test_i].cuda()
        num_node = graph_nnode[idx_test_i]
        y = graph_labels[idx_test_i].cuda()

        out, test_dict_loss, test_kl_loss = model(x, adj, num_node)
        # out, test_kl_loss = model(x, adj, num_node)

        # pred = out.max(dim=1)[1]
        # correct += pred.eq(y).sum().item()
        test_loss = cross_entropy(out, y) + test_dict_loss + test_kl_loss * args.kl_loss_ratio
        test_loss = test_loss.detach()
        # test_loss = cross_entropy(out, y) + test_kl_loss * args.kl_loss_ratio
        test_loss_list.append(test_loss)
        Calculate_acc_test.add(out, graph_labels[idx_test_i].cuda(), torch.tensor(labels_num))

    # acc_test = correct / len(dataset_test)
    acc_test = Calculate_acc_test.calculate_acc()
    return acc_test, test_loss_list


# 10-fold cross validation
idx_test_starts = []
for i in range(args.nfold):
    idx_test_starts.append(int(len(graphs) * i / args.nfold))
    if i == (args.nfold - 1):
        idx_test_starts.append(len(graphs))

np.random.seed(args.seed)
all_idx = np.arange(len(graphs))
np.random.shuffle(all_idx)

idx_all_nfold = []
for i in range(args.nfold):
    idx_all_nfold.append(all_idx[idx_test_starts[i]:idx_test_starts[i + 1]])

test_frequency = int(len(labels) / args.batch_size / args.test_fre)
accuracy_test_list = []
for ifold in range(0, args.nfold):
    lr = args.lr
    print('ifold:', ifold)
    accuracy_test_list_nfold = []

    # divide train set
    idx_test = idx_all_nfold[ifold]
    idx_train = np.array([], dtype=int)
    for i in range(args.nfold):
        if i != ifold:
            idx_train = np.concatenate((idx_train, idx_all_nfold[i]), axis=0)

    # construct graph dictionary
    dict_idx = queue_idx_each_class(labels_num, graph_labels, args.num_keys)
    dict_embs = [embs[k] for k in dict_idx]
    dict_adj = [graph_adj[k] for k in dict_idx]
    dict_maxnnode = get_node_max(dict_embs, args.dict_max_nnode)
    args.dict_max_nnode = dict_maxnnode
    dict_embs, dict_adj = get_node_unify(dict_embs, dict_adj, args.dimensionality, dict_maxnnode)

    dict_embs = torch.tensor(dict_embs)
    dict_labels = torch.tensor(graph_labels[dict_idx])
    dict_nnode = graph_nnode[torch.tensor(dict_idx)]
    graph_labels = torch.tensor(graph_labels)

    # construct model
    model = SS_GDE(args, dict_embs, dict_adj, dict_nnode, dict_labels).cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    # pdb.set_trace()
    # for para in model.named_parameters():
    #     print(para[0],'\t',para[1].size())
    # pdb.set_trace()
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda= lr_update_rule)
    # change cross-entroy loss to F.nll_loss
    cross_entropy = torch.nn.CrossEntropyLoss()

    for param_q, param_k in zip(model.encoder.parameters(), model.dict_encoder.parameters()):
        param_k.data.copy_(param_q.data.clone())  # initialize
        param_k.requires_grad = False  # not update by gradient
    t = time.time()
    best_test_acc = 0
    # training
    for epoch in range(args.epochs):

        if epoch % 40 == 0 and epoch != 0:
            lr = lr * 0.8
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                print('epoch:',epoch,' lr:',param_group['lr'])

        dataset_train = subDataset(idx_train)
        dataloader_train = DataLoader.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True,
                                                 num_workers=4,drop_last=False)

        dict_loss_list = []
        kl_loss_list = []
        train_loss_list = []
        correct = 0
        Calculate_acc_train = Calculate_acc()
        for img_num_i, data in enumerate(dataloader_train):

            if img_num_i % test_frequency == 0 and img_num_i != 0:
                acc_test, test_loss_list = test(idx_test)
                print('Epoch: {:04d}'.format(epoch + 1), 'acc_test: {:.6f}'.format(acc_test),
                      'best_acc_test: {:.6f}'.format(best_test_acc))

                if acc_test > best_test_acc:
                    best_test_acc = acc_test

            model.train()
            idx_train_i = data.cuda()

            # prepare training data
            x = embs_unify[idx_train_i].cuda()
            adj = adj_unify[idx_train_i].cuda()
            num_node = graph_nnode[idx_train_i]
            y = graph_labels[idx_train_i].cuda()

            # train model
            out, dict_loss, kl_loss = model(x, adj, num_node)
            # out, kl_loss = model(x, adj, num_node)

            Calculate_acc_train.add(out, graph_labels[idx_train_i].cuda(), torch.tensor(labels_num))

            # pred = out.max(dim=1)[1]
            # correct += pred.eq(y).sum().item()
            loss = cross_entropy(out, y)

            kl_loss_list.append(args.kl_loss_ratio * kl_loss)
            dict_loss_list.append(dict_loss)
            train_loss_list.append(loss)

            train_loss = loss + dict_loss + args.kl_loss_ratio * kl_loss
            # train_loss = loss + args.kl_loss_ratio * kl_loss

            # gradient backward
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            for param_q, param_k in zip(model.encoder.parameters(), model.dict_encoder.parameters()):
                param_k.data = param_k.data * args.m + param_q.data * (1. - args.m)

        acc_train = Calculate_acc_train.calculate_acc()
        # acc_train = correct / len(dataset_train)
        # scheduler.step()

        acc_test, test_loss_list = test(idx_test)
        if acc_test > best_test_acc:
            best_test_acc = acc_test

        print('Epoch: {:04d}'.format(epoch + 1), 'kl_loss:{:.4f}'.format(torch.tensor(kl_loss_list).mean()),
              'dict_loss:{:.4f}'.format(torch.tensor(dict_loss_list).mean()),
              'train_loss: {:.4f}'.format(torch.tensor(train_loss_list).mean()),
              'acc_train: {:.6f}'.format(acc_train), 'acc_test: {:.6f}'.format(acc_test),
              'best_acc_test: {:.6f}'.format(best_test_acc),
              'time: {:.6f}s'.format(time.time() - t))

    accuracy_test_list.append(best_test_acc)
    print('10-fold accuracies: ', accuracy_test_list)

accuracy_test = torch.tensor(accuracy_test_list)
print("acc_test", accuracy_test)
print("Optimization Finished!")
print("nfold: mean and std:", accuracy_test.mean().data, accuracy_test.std().data)