from mha import MHA, LegacyMHA, project_W
from data.generate_synth_data import generate_all
from argparse import ArgumentParser
from metrics import cluster_score, covariance_mse
import pickle
import numpy as np
import tqdm


def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-N', type=int, default=1, help='number of subjects')
    parser.add_argument('-n', type=int, default=10,
                        help='number of observations')
    parser.add_argument('-k', type=int, default=5, help='latent dim')
    parser.add_argument('-p', type=int, default=50, help='observation dim')
    parser.add_argument('-i', '--n-iters', type=int,
                        default=250, help='number of iterations')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    n = args.n
    N = args.N
    p = args.p
    k = args.k
    n_iters = args.n_iters

    res = {'jaccard': [], 'hamming': [], 'kulsinski': [],
           'score': [], 'cov_mse': {i: [] for i in range(N)}}
    res_leg = {'jaccard': [], 'hamming': [], 'kulsinski': [],
               'score': [], 'cov_mse': {i: [] for i in range(N)}}

    distances = ['jaccard', 'hamming', 'kulsinski']

    print(f"Running for {N} subjects and {n} observations")
    for iteration in tqdm.tqdm(range(n_iters)):
        X, Z, W, G = generate_all(n, N, p, k, seed=iteration)
        covs = [np.cov(x, rowvar=False) for x in X]

        model = MHA(k=k)
        model.fit(X=X, max_iter=5000)
        cs = cluster_score(model.W, W)
        for d in distances:
            res[d].append(cs[0][d])
        res['score'].append(cs[1])
        for in in range(N):
            res['cov_mse'][i].append(covariance_mse(model.G[i], G[i]))

        model_leg = LegacyMHA(covs, k=k)
        model_leg.fit(maxIter=5000)
        cs = cluster_score(model_leg.W, W)
        for d in distances:
            res_leg[d].append(cs[0][d])
        res_leg['score'].append(cs[1])
        for in in range(N):
            res_leg['cov_mse'][i].append(covariance_mse(model_leg.G[i], G[i]))

    pickle.dump(res,
                open(f'results/results_{N}_{n}_{p}_{k}_{n_iters}.p', 'wb'))
    pickle.dump(res_leg,
                open(f'results/results_leg_{N}_{n}_{p}_{k}_{n_iters}.p', 'wb'))
