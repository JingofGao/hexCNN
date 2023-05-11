import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf

from copy import deepcopy
from torch_scatter import scatter
from torch_geometric.nn.conv import GCNConv, GATConv, GINConv


class GatedLinearUnit(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        xx = x.chunk(2, 1)
        xx = nnf.relu(xx[0]) * xx[1]
        return xx

class GCN(nn.Module):
    def __init__(self, dim_input, width, depth):
        super().__init__()

        self.foot = nn.Sequential(nn.Dropout(),
                        nn.Conv1d(dim_input, width, 1),
                        nn.GELU())
        self.dropout = nn.Dropout()
        block = GCNConv(width, width)
        self.block = nn.ModuleList([deepcopy(block) for _ in range(depth+1)])
        self.head = nn.Sequential(nn.GELU(),
                        nn.Conv1d(width, 7, 1))
        print('##params[model]:', np.sum([np.prod(p.shape) for p in self.parameters()]))
        print()

    def forward(self, graph):
        xx = self.foot(graph.x.unsqueeze(-1)).squeeze(-1)
        for b in self.block:
            xx = xx + b(self.dropout(xx), graph.edge_index)
        xx = self.head(xx.unsqueeze(-1)).squeeze(-1)
        xx = xx[graph.center+graph.ptr[:-1]]
        return xx

class GAT(nn.Module):
    def __init__(self, dim_input, width, depth):
        super().__init__()

        self.foot = nn.Sequential(nn.Dropout(),
                        nn.Conv1d(dim_input, width, 1),
                        nn.GELU())
        self.dropout = nn.Dropout()
        block = GATConv(width, width)
        self.block = nn.ModuleList([deepcopy(block) for _ in range(depth+1)])
        self.head = nn.Sequential(nn.GELU(),
                        nn.Conv1d(width, 7, 1))
        print('##params[model]:', np.sum([np.prod(p.shape) for p in self.parameters()]))
        print()

    def forward(self, graph):
        xx = self.foot(graph.x.unsqueeze(-1)).squeeze(-1)
        for b in self.block:
            xx = xx + b(self.dropout(xx), graph.edge_index)
        xx = self.head(xx.unsqueeze(-1)).squeeze(-1)
        xx = xx[graph.center+graph.ptr[:-1]]
        return xx

class GIN(nn.Module):
    def __init__(self, dim_input, width, depth):
        super().__init__()

        self.foot = nn.Sequential(nn.Dropout(),
                        nn.Conv1d(dim_input, width, 1),
                        nn.GELU())
        block = GINConv(nn.Sequential(nn.Dropout(),
                    nn.Linear(width, width),
                    nn.GELU(),
                    nn.Linear(width, width)))
        self.block = nn.ModuleList([deepcopy(block) for _ in range(depth+1)])
        self.head = nn.Sequential(nn.GELU(),
                        nn.Conv1d(width, 7, 1))
        print('##params[model]:', np.sum([np.prod(p.shape) for p in self.parameters()]))
        print()

    def forward(self, graph):
        xx = self.foot(graph.x.unsqueeze(-1)).squeeze(-1)
        for b in self.block:
            xx = xx + b(xx, graph.edge_index)
        xx = self.head(xx.unsqueeze(-1)).squeeze(-1)
        xx = xx[graph.center+graph.ptr[:-1]]
        return xx


class BaseBlock(nn.Module):
    def __init__(self, width, submodel=None):
        super().__init__()

        numgrp = width // 64
        self.pre = nn.Sequential(nn.GroupNorm(numgrp, width, affine=False),
                        nn.Conv1d(width, width*8, 1, groups=numgrp),
                        GatedLinearUnit())
        if submodel is None:
            self.premodel = self.submodel = None
        else:
            premodel = nn.Sequential(nn.Dropout(),
                           nn.Conv1d(width, width, 1))
            self.premodel = nn.ModuleList([deepcopy(premodel) for _ in range(2)])
            self.submodel = submodel
        self.post = nn.Sequential(nn.Dropout(),
                        nn.Conv1d(width*4, width, 1, groups=numgrp))
        print('##params[base]:', np.sum([np.prod(p.shape) for p in self.parameters()]))

    def subforward(self, x, i, edge_index, edge_attr):
        xx = self.premodel[i](x)
        xx = self.submodel(xx, edge_index, edge_attr)
        return x + xx

    def forward(self, x, edge_index, edge_attr):
        xx = self.pre(x)
        if self.submodel is not None:
            x0, x1, x2, x3 = xx.chunk(4, 1)
            x0 = self.subforward(x0, 0, edge_index, edge_attr)
            x1 = self.subforward(x1, 1, edge_index, edge_attr)
            xx = pt.cat([x0, x1, x2, x3], 1)
            x0 = x1 = x2 = x3 = None
        xx = self.post(xx)
        return xx

class ConvBlock(nn.Module):
    def __init__(self, model_mode, conv_mode, width, submodel=None):
        super().__init__()
        if model_mode == 4:
            self.conv_size = 1
        elif model_mode == 5:
            self.conv_size = conv_mode * 6 + 1
        else:
            raise Exception('Unknown model_mode:', model_mode)

        numgrp = width // 64
        self.pre = nn.Sequential(nn.GroupNorm(numgrp, width, affine=False),
                        nn.Conv1d(width, width*8, 1, groups=numgrp),
                        GatedLinearUnit())
        if submodel is None:
            self.premodel = self.submodel = None
        else:
            premodel = nn.Sequential(nn.Dropout(),
                           nn.Conv1d(self.conv_size*width, self.conv_size*width, 1, groups=self.conv_size))
            self.premodel = nn.ModuleList([deepcopy(premodel) for _ in range(2)])
            self.submodel = submodel
        self.post = nn.Sequential(nn.Dropout(),
                        nn.Conv1d(width*4, width, 1, groups=numgrp))
        print('##params[conv%d]:' % self.conv_size, np.sum([np.prod(p.shape) for p in self.parameters()]))

    def subforward(self, x, i, edge_index, edge_attr):
        xx = x.unsqueeze(1).expand(-1, self.conv_size, -1, -1).reshape(x.shape[0], -1, x.shape[2])
        xx = self.premodel[i](xx).reshape(x.shape[0], self.conv_size, -1, x.shape[2])
        if self.conv_size == 1:
            msg = xx[edge_index[0], 0]
        else:
            msg = xx[edge_index[0], edge_attr[:, 0]]
        msg = scatter(msg, edge_index[1], dim=0, dim_size=len(x), reduce='mean')
        xx, msg = xx[:, -1] + msg, None
        xx = self.submodel(xx, edge_index, edge_attr)
        return x + xx

    def forward(self, x, edge_index, edge_attr):
        xx = self.pre(x)
        if self.submodel is not None:
            x0, x1, x2, x3 = xx.chunk(4, 1)
            x0 = self.subforward(x0, 0, edge_index, edge_attr)
            x1 = self.subforward(x1, 1, edge_index, edge_attr)
            xx = pt.cat([x0, x1, x2, x3], 1)
            x0 = x1 = x2 = x3 = None
        xx = self.post(xx)
        return xx


class ContainerModel(nn.Module):
    def __init__(self, model_mode, conv_mode, dim_input, width, depth):
        super().__init__()

        self.foot = nn.Sequential(nn.Dropout(),
                        nn.Conv1d(dim_input, width, 1),
                        nn.GELU())
        self.premodel = nn.Sequential(nn.Dropout(),
                            nn.Conv1d(width, width, 1))
        submodel = BaseBlock(width, None)
        if model_mode == 0:
            for _ in range(depth):
                submodel = BaseBlock(width, submodel)
        else:
            for _ in range(depth):
                submodel = ConvBlock(model_mode, conv_mode, width, submodel)
        self.submodel = submodel
        self.head = nn.Sequential(nn.GELU(),
                        nn.Conv1d(width, 7, 1))
        print('##params[model]:', np.sum([np.prod(p.shape) for p in self.parameters()]))
        print()

    def forward(self, graph):
        xx = self.foot(graph.x.unsqueeze(-1))
        xx = self.premodel(xx)
        xx = self.submodel(xx, graph.edge_index, graph.edge_attr)
        xx = xx[graph.center+graph.ptr[:-1]]
        xx = self.head(xx).squeeze(-1)
        return xx

