# data process
# before: (59904, 33538)
# after: (59904, 12848)

import os
import h5py
import numpy as np
import scanpy as sp

from glob import glob
import argparse


def main(args):
    alphabet = ['WM', 'Layer_1', 'Layer_2', 'Layer_3', 'Layer_4', 'Layer_5', 'Layer_6']

    didx, dy, dx, dpos = [], [], [], []
    for idx, fdn in enumerate(sorted(glob('{}/*'.format(args.input_path)))):
        fbn = os.path.basename(fdn)
        print('#loading', fbn, '...')

        label = {}
        with open(fdn + '/' + fbn + '_truth.txt', 'r') as f:
            for l in f.readlines():
                try: k, v = l.split()
                except: continue
                label[k] = alphabet.index(v)
        print('#label:', len(label))

        data = sp.read_visium(path=fdn, count_file=fbn+'_raw_feature_bc_matrix.h5')
        # data.var_names_make_unique()
        coord = data.obsm['spatial']
        feat = data.X  # dist: 140, 240, 280
        tag = data.obs.index
        print('#data:', feat.shape, coord.shape, len(tag))

        for ci, xi, yi in zip(coord, feat, tag):
            didx.append(idx)
            dpos.append(ci[None, :])
            dx.append(xi.toarray())
            dy.append(label.get(yi, -1))
        print()
    didx = np.array(didx).astype(np.int8)
    dpos = np.concatenate(dpos, axis=0).astype(np.int16)
    dx = np.concatenate(dx, axis=0).astype(np.int16)
    dy = np.array(dy).astype(np.int8)

    cutoff = [[0.01, '001']]
    for cnum, cstr in cutoff:
        msk = np.mean(dx>0, axis=0) > cnum
        with h5py.File('{}/mask{}.hdf5'.format(args.output_path, cstr), 'w') as f:
            f.create_dataset('index', data=msk)
        print('#before:', dx.shape, cnum)
        xx = dx[:, msk]
        print('#after:', xx.shape, cnum)

        with h5py.File('{}/data{}.hdf5'.format(args.output_path,cstr), 'w') as f:
            f.create_dataset('index', data=didx)
            f.create_dataset('coord', data=dpos)
            f.create_dataset('x', data=xx)
            f.create_dataset('y', data=dy)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_path', type=str, default="data/raw")
    parser.add_argument('--output_path', type=str, default="data/process", )
    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    main(args)
