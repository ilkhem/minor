from mha import MHA, LegacyMHA, project_W
from data.generate_synth_data import generate_all
from argparse import ArgumentParser
from metrics import cluster_score
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

    res = {N : {n: [] for n in n_vals} for N in N_vals}
    res_leg = {N : {n: [] for n in n_vals} for N in N_vals}

    for N in N_vals:
        for n in n_vals:
            print(f"Running for {N} subjects and {n} observations")
            for iteration in tqdm.tqdm(range(n_iters)):
                X, Z, W, G = generate_all(n, N, p, k, ones=True, seed=iteration)
                covs = [np.cov(x, rowvar=False) for x in X]

                model = MHA(k=k)
                model.fit(X=X)
                Wm = project_W(model.W, ones=True)
                res[N][n].append(cluster_score(Wm, W)[2])

                model_leg = LegacyMHA(covs, k=k)
                model_leg.fit()
                Wml = project_W(model_leg.W, ones=True)
                res_leg[N][n].append(cluster_score(Wml, W)[2])


    pickle.dump(res, open('results.p', 'wb'))
    pickle.dump(res_leg, open('results_leg.p', 'wb'))

