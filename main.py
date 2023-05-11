#!/home/xfcui/miniconda38/bin/python -Bu

import os
import argparse
import numpy as np
import torch as pt
import torch.nn as nn
import torch.nn.functional as nnf

from tqdm import tqdm
from copy import deepcopy
from torch_geometric.loader import DataLoader
from torch.optim import SGD, AdamW

from data import SpatialDataset, transform
from model import ContainerModel, GCN, GAT, GIN


def main(args):
    total_acc = []
    for f in range(args.fold):
        dataset = SpatialDataset(args.input_filename, max(args.model_depth-1, 1))
        trainset, testset = dataset.split(f, args.conv_mode)
        dataset = None
        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, drop_last=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, drop_last=False)
        print('#batch:', batch_size)
        print()

        result = 0
        ensemble_acc = []
        print('#training ...')
        for idx in range(args.ensemble_num):
            ofn = '{}/{}_f{}_e{}.pt'.format(args.output_filename, args.model_name, f, idx)
            if args.model_name == "GCN":
                model = GCN(trainset.getDimX(), args.model_width, args.model_depth).cuda()
            elif args.model_name == "GAT":
                model = GAT(trainset.getDimX(), args.model_width, args.model_depth).cuda()
            elif args.model_name == "MLP":
                model = ContainerModel(0, args.conv_mode, trainset.getDimX(), args.model_width,
                                       args.model_depth).cuda()
            else:
                model = ContainerModel(5, args.conv_mode, trainset.getDimX(), args.model_width,
                                       args.model_depth).cuda()

            model.load_state_dict(pt.load(ofn))

            y_lst = []
            model.eval()
            for graph in tqdm(testloader):
                graph = transform(graph, False, args.conv_mode)

                y = nnf.softmax(model(graph), -1)
                y_lst.append(y.detach().cpu())
            result += pt.cat(y_lst)
            acc = pt.mean((pt.argmax(result, dim=-1) == testset.y[testset.valid]).float()) * 100.0
            ensemble_acc.append(acc)
        total_acc.append(np.mean(ensemble_acc))
    print("test acc: %.3f" % np.mean(total_acc))


if __name__ == '__main__':
    models = ["MLP", "GCN", "GAT", "hexCNN"]

    parser = argparse.ArgumentParser(usage='Usage: %(prog)s [options]')
    parser.add_argument('-i', '--input', dest='input_filename', default='data/process/data001.hdf5')
    parser.add_argument('-o', '--output', dest='output_filename', default='output')
    parser.add_argument('-f', '--fold', dest='fold', default=4, type=int)
    parser.add_argument('-a', '--augment', dest='augment', default=0, type=int)
    parser.add_argument('-m', '--model', dest='model_name', default="hexCNN",
                        type=str, choices=models)
    parser.add_argument('-c', '--conv', dest='conv_mode', default=2, type=int)  # topology = conv_mode*2+1 = 1, 7, 13
    parser.add_argument('-d', '--depth', dest='model_depth', default=2, type=int)
    parser.add_argument('-w', '--width', dest='model_width', default=768, type=int)
    parser.add_argument('-e', '--ensemble', dest='ensemble_num', default=1, type=int)
    parser.add_argument('-p', '--patience', dest='patience_num', default=12, type=int)
    args = parser.parse_args()
    print('#config:', args)
    print()

    batch_size, learn_rate, weight_decay = 192, 1e-4, 10

    if not os.path.exists(args.output_filename):
        os.makedirs(args.output_filename)

    main(args)