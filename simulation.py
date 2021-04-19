import os
import sys
import time
import pickle
from argparse import ArgumentParser
import numpy as np
from tqdm import tqdm
from mha import MHA, LegacyMHA, project_W
from data.generate_synth_data import generate_all
from metrics import get_alignment, distances


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

    keys = ['jaccard', 'hamming', 'kulsinski', 'score', 'ha_jaccard', 'ha_hamming',
            'ha_kulsinski', 'ha_score', 'cov_mse', 'cov_mse_mean', 'cov_mse_max',
            'ha_cov_mse', 'ha_cov_mse_mean', 'ha_cov_mse_max', 'run_time']
    res = {key: [] for key in keys}
    res_leg = {key: [] for key in keys}

    print(f"Running for {N} subjects and {n} observations")
    for iteration in tqdm(range(n_iters)):
        X, Z, W, G = generate_all(n, N, p, k, seed=iteration)
        covs = [np.cov(x, rowvar=False) for x in X]

        st = time.time()
        model = MHA(k=k)
        model.fit(X=X, max_iter=5000, init_method='sparse_svd')
        duration = time.time() - st
        res['run_time'].append(duration)
        # align according to eucl distance
        alignment, score = get_alignment(model.W, W)
        dists = distances(model.W, W, model.G, G, alignment=alignment)
        res['score'].append(score)
        for dist_key, dist_value in dists.items():
            res[dist_key].append(dist_value)
        # align according to hamming distance
        alignment, score = get_alignment(model.W, W, cost_dist='hamming')
        dists = distances(model.W, W, model.G, G, alignment=alignment)
        res['ha_score'].append(score)
        for dist_key, dist_value in dists.items():
            res[f'ha_{dist_key}'].append(dist_value)

        st = time.time()
        model_leg = LegacyMHA(covs, k=k)
        model_leg.fit(maxIter=5000)
        duration = time.time() - st
        res_leg['run_time'].append(duration)
        # align according to eucl distance
        alignment, score = get_alignment(model.W, W)
        dists = distances(model.W, W, model.G, G, alignment=alignment)
        res_leg['score'].append(score)
        for dist_key, dist_value in dists.items():
            res_leg[dist_key].append(dist_value)
        # align according to hamming distance
        alignment, score = get_alignment(model.W, W, cost_dist='hamming')
        dists = distances(model.W, W, model.G, G, alignment=alignment)
        res_leg['ha_score'].append(score)
        for dist_key, dist_value in dists.items():
            res_leg[f'ha_{dist_key}'].append(dist_value)

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
