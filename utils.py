import math
import torch
import numpy as np


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