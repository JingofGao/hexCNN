import h5py
import random
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf
import faiss
import faiss.contrib.torch_utils

from copy import deepcopy
from torch_geometric.utils import k_hop_subgraph, dropout_adj
from torch_geometric.data import Data, InMemoryDataset

seed = 20201212
random.seed(seed)
np.random.seed(seed)
pt.manual_seed(seed)


class SpatialDataset(InMemoryDataset):
    def __init__(self, filename, hop):
        self.hop = hop
        if filename is None: return
        print('#loading', filename, '...')

        with h5py.File(filename) as f:
            i = f['index'][()]
            c = f['coord'][()]
            x = f['x'][()]
            y = f['y'][()]

        self.i = pt.from_numpy(i).short()
        self.c = pt.from_numpy(c).float()
        self.x = pt.from_numpy(x).float()
        self.y = pt.from_numpy(y).long()
        self.valid = pt.where(self.y >= 0)[0]
        print('#size:', len(self.y), len(self))

    def build(self, conv_mode):
        self.cache = [None] * len(self.valid)

        x = pt.cat([self.c, self.i.unsqueeze(-1)*1000], dim=-1).float()
        index = faiss.IndexFlatL2(3)
        index.add(x)
        dist, knn = index.search(x, 13)
        knn[dist > np.square(244)] = -1
        self.edge_index = pt.cat([knn[:, [i, 0]] for i in range(1, 13)])
        self.edge_index = self.edge_index[self.edge_index[:, 0]>=0].T

        center = [[  -1.06,  239.84],
                  [  68.39,  120.23],
                  [ 206.23,  120.85],
                  [ 137.84,    0.61],
                  [ 207.3 , -119.01],
                  [  69.45, -119.61],
                  [   1.05, -239.86],
                  [ -68.39, -120.23],
                  [-206.24, -120.81],
                  [-137.85,   -0.61],
                  [-207.3 ,  119.01],
                  [ -69.46,  119.63]]
        center = pt.from_numpy(np.array(center)).float()
        index = faiss.IndexFlatL2(2)
        index.add(center)
        coord = self.c[self.edge_index[1, :]] - self.c[self.edge_index[0, :]]
        _, self.edge_attr = index.search(coord, 1)

        if conv_mode == 0:
            self.edge_index = pt.zeros([2, 0])
            self.edge_attr = pt.zeros([1, 0])
        elif conv_mode == 1:
            idx = ((self.edge_attr % 2) == 1)[:, 0]
            self.edge_index = self.edge_index[:, idx]
            self.edge_attr = pt.div(self.edge_attr[idx], 2, rounding_mode='floor')
        elif conv_mode == 2:
            pass
        else:
            raise Exception('Unknown conv_mode:', conv_mode)

    def split(self, fold, conv_mode):
        if fold is None:
            idx = pt.rand(self.y.shape) < 0.25
            trainset = SpatialDataset(None, self.hop)
            trainset.i = self.i
            trainset.c = self.c
            trainset.x = self.x
            trainset.y = deepcopy(self.y); trainset.y[idx] = -1
            trainset.valid = pt.where(trainset.y >= 0)[0]

            idx = pt.logical_not(idx)
            testset = SpatialDataset(None, self.hop)
            testset.i = self.i
            testset.c = self.c
            testset.x = self.x
            testset.y = deepcopy(self.y); testset.y[idx] = -1
            testset.valid = pt.where(testset.y >= 0)[0]

            print('#split[rnd]:', len(trainset), len(testset))
        else:
            idx = (self.i % 4) != fold
            trainset = SpatialDataset(None, self.hop)
            trainset.i = self.i[idx]
            trainset.c = self.c[idx]
            trainset.x = self.x[idx]
            trainset.y = self.y[idx]
            trainset.valid = pt.where(trainset.y >= 0)[0]

            idx = pt.logical_not(idx)
            testset = SpatialDataset(None, self.hop)
            testset.i = self.i[idx]
            testset.c = self.c[idx]
            testset.x = self.x[idx]
            testset.y = self.y[idx]
            testset.valid = pt.where(testset.y >= 0)[0]

            print('#split[%d]:' % fold, len(trainset), len(testset))

        trainset.build(conv_mode)
        testset.build(conv_mode)
        return trainset, testset

    def getDimX(self):
        return self.x[0].shape[-1]

    def __len__(self):
        return len(self.valid)

    def __getitem__(self, idx):
        if self.cache[idx] is None:
            i = self.valid[idx].item()
            if self.edge_index.shape[-1] > 0:
                node_mask, edge_index, center, edge_mask = k_hop_subgraph(i, self.hop, self.edge_index, relabel_nodes=True)
                node_attr, edge_attr = self.x[node_mask], self.edge_attr[edge_mask]
                label = self.y[i:i+1].clone()
                self.cache[idx] = Data(node_attr, edge_index, edge_attr, label, center=center)
            else:
                node_attr, label = self.x[i:i+1].clone(), self.y[i:i+1].clone()
                self.cache[idx] = Data(node_attr, self.edge_index, self.edge_attr, label, center=pt.zeros(1).long())
        return deepcopy(self.cache[idx])

def transform(graph, augment, conv_mode):
    graph = graph.cuda()

    if augment:
        if conv_mode == 0:
            pass
        elif conv_mode == 1:
            rnd = pt.randint(12, [1]).item()
            if rnd < 6:
                graph.edge_attr = (rnd + graph.edge_attr) % 6
            else:
                graph.edge_attr = (rnd - graph.edge_attr) % 6
            graph.edge_index, graph.edge_attr = dropout_adj(graph.edge_index, graph.edge_attr, 1/6, force_undirected=True)
        elif conv_mode == 2:
            rnd = pt.randint(12, [1]).item() * 2
            if rnd < 12:
                graph.edge_attr = (rnd + graph.edge_attr) % 12
            else:
                graph.edge_attr = (rnd - graph.edge_attr) % 12
            graph.edge_index, graph.edge_attr = dropout_adj(graph.edge_index, graph.edge_attr, 1/6, force_undirected=True)
        else:
            raise Exception('Unknown conv_mode:', conv_mode)

    return graph

