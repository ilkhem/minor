from mha import MHA, LegacyMHA, project_W
from data.generate_synth_data import generate_all
from argparse import ArgumentParser
from metrics import cluster_score, covariance_mse
import pickle
import numpy as np
import tqdm

def parse_arguments():
    parser = ArgumentParser()
    parser.add_argument('-k', type=int, default=5, help='latent dim')
    parser.add_argument('-p', type=int, default=10, help='observation dim')
    parser.add_argument('-i', '--n-iters', type=int, default=250, help='number of iterations')
    return parser.parse_args()



if __name__ == "__main__":
    args = parse_arguments()
    n = 1000
    N = 1
    p = args.p
    k = args.k
    n_iters = args.n_iters

    N_vals = [1, 2, 3, 5, 10]
    n_vals = [10, 25, 50, 100, 150, 250, 500, 1000, 1500, 2500]
    distances = ['jaccard', 'hamming', 'kulsinski']

    res = {N : {n: {d: [] for d in distances} for n in n_vals} for N in N_vals}
    res_leg = {N : {n: {d: [] for d in distances} for n in n_vals} for N in N_vals}

    for N in N_vals:
        for n in n_vals:
            print(f"Running for {N} subjects and {n} observations")
            for iteration in tqdm.tqdm(range(n_iters)):
                X, Z, W, G = generate_all(n, N, p, k, ones=True, seed=iteration)
                covs = [np.cov(x, rowvar=False) for x in X]

                model = MHA(k=k)
                model.fit(X=X, max_iter=5000)
                for d in distances:
                    res[N][n][d].append(cluster_score(model.W, W, distance=d))
                res[N][n]['cov_mse'] = []
                for i in range(N):
                    res[N][n]['cov_mse'].append(covariance_mse(model.G[i], G[i]))

                model_leg = LegacyMHA(covs, k=k)
                model_leg.fit(maxIter=5000)
                for d in distances:
                    res_leg[N][n][d].append(cluster_score(model_leg.W, W, distance=d))
                res_leg[N][n]['cov_mse'] = []
                for i in range(N):
                    res_leg[N][n]['cov_mse'].append(covariance_mse(model_leg.G[i], G[i]))

    pickle.dump(res, open(f'results_{n_iters}.p', 'wb'))
    pickle.dump(res_leg, open(f'results_leg_{n_iters}.p', 'wb'))

