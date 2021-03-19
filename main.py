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


def parse_dtype(dtype):
    if dtype == 'float32':
        return np.float32
    elif dtype == 'float64':
        return np.float64
    else:
        raise ValueError(f"illegal dtype: {dtype}")

def _delist_single(l):
    return l[0] if len(l) == 1 else l

def main():
    args = parse_arguments()
    k = args.k
    run_dir = args.run
    dtype = parse_dtype(args.dtype)
    args.sub = _delist_single(args.sub)
    args.ses = _delist_single(args.ses)

    print("subjects", _check_idx(args.sub))
    print("sessions", _check_idx(args.ses))
    print(type(_check_idx(args.ses)))
    # os.makedirs(run_dir, exist_ok=True)


if __name__ == "__main__":
    sys.exit(main())
