import os
import sys
import time
import pickle
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from mha import MHA, LegacyMHA, project_W
from data.generate_synth_data import generate_all
from metrics import cluster_score, covariance_mse


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='number of subjects')
    parser.add_argument('-n', type=int, default=10,
                        help='number of observations')
    parser.add_argument('-k', type=int, default=5, help='latent dim')
    parser.add_argument('-p', type=int, default=50, help='observation dim')
    parser.add_argument('-i', '--n-iters', type=int,
                        default=250, help='number of iterations')
    parser.add_argument('-r', '--run', type=str,
                        default='results', help='Dir to save results')
    parser.add_argument('-t', '--time', action='store_true',
                        help='run time complexity')
    return parser.parse_args()


def main_sim(N, n, p, k, n_iters, run_dir):

    res = {'jaccard': [], 'hamming': [], 'kulsinski': [], 'score': [],
           'ha_jaccard': [], 'ha_hamming': [], 'ha_kulsinski': [], 'ha_score': [],
           'cov_mse': {i: [] for i in range(N)}}
    res_leg = {'jaccard': [], 'hamming': [], 'kulsinski': [], 'score': [],
               'ha_jaccard': [], 'ha_hamming': [], 'ha_kulsinski': [], 'ha_score': [],
               'cov_mse': {i: [] for i in range(N)}}

    distances = ['jaccard', 'hamming', 'kulsinski']

    print(f"Running for {N} subjects and {n} observations")
    for iteration in tqdm(range(n_iters)):
        X, Z, W, G = generate_all(n, N, p, k, seed=iteration)
        covs = [np.cov(x, rowvar=False) for x in X]

        model = MHA(k=k)
        model.fit(X=X, max_iter=5000, init_method='sparse_svd')
        cs = cluster_score(model.W, W)
        for d in distances:
            res[d].append(cs[0][d])
        res['score'].append(cs[1])
        cs = cluster_score(model.W, W, cost_dist='hamming')
        for d in distances:
            res[f'ha_{d}'].append(cs[0][d])
        res['ha_score'].append(cs[1])
        for i in range(N):
            res['cov_mse'][i].append(covariance_mse(model.G[i], G[i]))

        model_leg = LegacyMHA(covs, k=k)
        model_leg.fit(maxIter=5000)
        cs = cluster_score(model_leg.W, W)
        for d in distances:
            res_leg[d].append(cs[0][d])
        res_leg['score'].append(cs[1])
        cs = cluster_score(model_leg.W, W, cost_dist='hamming')
        for d in distances:
            res_leg[f'ha_{d}'].append(cs[0][d])
        res_leg['ha_score'].append(cs[1])
        for i in range(N):
            res_leg['cov_mse'][i].append(covariance_mse(model_leg.G[i], G[i]))

    pickle.dump(res,
                open(f'{run_dir}/results_{N}_{n}_{p}_{k}_{n_iters}.p', 'wb'))
    pickle.dump(res_leg,
                open(f'{run_dir}/results_leg_{N}_{n}_{p}_{k}_{n_iters}.p', 'wb'))


def time_complexity(n, p, k, n_iters, run_dir):
    N_vals = range(1, 21)
    res = {N: [] for N in N_vals}

    for N in N_vals:
        for it in tqdm(range(n_iters)):
            X, Z, W, G = generate_all(n, N, p, k, seed=it)
            st = time.time()
            model = MHA(k=k)
            model.fit(X=X, max_iter=5000, init_method='sparse_svd')
            duration = time.time() - st
            res[N].append(duration)

    pickle.dump(res, open(f'{run_dir}/time_{n}_{p}_{k}_{n_iters}.p', 'wb'))


if __name__ == "__main__":
    args = parse_arguments()
    n = args.n
    N = args.N
    p = args.p
    k = args.k
    n_iters = args.n_iters
    run_dir = args.run
    os.makedirs(run_dir, exist_ok=True)
    rt = args.time
    if not rt:
        sys.exit(main_sim(N, n, p, k, n_iters, run_dir))
    else:
        sys.exit(time_complexity(n, p, k, n_iters, run_dir))
