import os
import sys
import time
import pickle
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from data.fmri import load, _check_idx
from mha import MHA


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-k', type=int, default=5, help='latent dim')
    parser.add_argument('-s', '--sub', default=1,
                        nargs='*', help='Subjects to load')
    parser.add_argument('-e', '--ses', default=1, nargs='*',
                        help='Sessions to load for each subject')
    parser.add_argument('-i', '--init-method',
                        default='random_svd', type=str, help='Init. method')
    parser.add_argument('-v', '--verbose', action='store_true')
    parser.add_argument('-d', '--dtype', default='float32',
                        type=str, help='Data type, e.g. float32, float64...')
    parser.add_argument('-r', '--run', type=str,
                        default='results', help='Dir to save results')
    return parser.parse_args()


def _parse_dtype(dtype):
    if dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    else:
        raise ValueError(f"illegal dtype: {dtype}")


def _delist_single(l):
    return l[0] if len(l) == 1 else l


def _format_idx(idx):
    idx = _check_idx(idx)
    return ''.join([str(int(i) - 1) for i in idx])


def _format_im(im):
    idict = {'random_svd': 'rsvd', 'random': 'r',
             'sparse_svd': 'svds', 'truncated_svd': 'tsvd'}
    return idict[im]


def main():
    args = parse_arguments()
    k = args.k
    v = args.verbose
    run_dir = args.run
    dtype = _parse_dtype(args.dtype)
    sub_idx = _delist_single(args.sub)
    ps = _format_idx(sub_idx)
    ses_idx = _delist_single(args.ses)
    pe = _format_idx(ses_idx)
    im = args.init_method
    pim = _format_im(im)
    max_iter = 30000

    data = load(sub_idx=sub_idx, ses_idx=ses_idx, dtype=dtype)
    model = MHA(k=k)
    model.fit(data, verbose=v, init_method=im, max_iter=max_iter)

    pn = '' if model.n_iters < max_iter else 'c'

    os.makedirs(run_dir, exist_ok=True)
    print("Training done. Saving matrices...")
    pickle.dump({'W': model.W, 'G': model.G, 'n_iters': model.n_iters},
                open(f'{run_dir}/par_{pim}_k{k}_s{ps}_e{pe}{pn}.p', 'wb'))


if __name__ == "__main__":
    sys.exit(main())
