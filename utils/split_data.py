from __future__ import print_function

import numpy as np
import argparse
import os

def split_data(x, y, outdir='', train_size=.85, val_size=.10):
    np.random.seed(7)

    n_samples = len(x)
    n_train = int(n_samples*train_size)
    n_val = int(n_samples*val_size)
    n_test = n_samples - (n_train + n_val)

    indices = np.random.permutation(n_samples)

    train_ind = indices[:n_train]
    val_ind = indices[n_train:n_train+n_val]
    test_ind = indices[n_val:]

    np.save(outdir+'train.npy', x[train_ind])
    np.save(outdir+'val.npy', x[val_ind])
    np.save(outdir+'test.npy', x[test_ind])

    np.save(outdir+'train_labels.npy', y[train_ind])
    np.save(outdir+'val_labels.npy', y[val_ind])
    np.save(outdir+'test_labels.npy', y[test_ind])

    if outdir == '':
        outdir = 'current working directory'

    print('Saved {} Train, {} Val, and {} Test examples to {}'.format(n_train, n_val, n_test, outdir))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('samples', type=str)
    parser.add_argument('labels', type=str)
    parser.add_argument('--outdir', default='')

    args = parser.parse_args()

    x = np.load(args.samples)
    y = np.load(args.labels)

    if not os.path.isdir(args.outdir):
        os.makedirs(args.outdir)

    split_data(x, y, outdir=args.outdir)