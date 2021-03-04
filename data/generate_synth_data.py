"""
Generate synthetic data that follows the MHA model, in
which Z_i ~ Norm(0, G_i), X_i ~ Norm(WZ_i, I)

Notations:
    N: number of subjects
    p: dimension of the observation X
    k: dimension of the latent space Z
    n: number of observations
    G_i: covariance matrix for i-th subject
    W: loading matrix shared across all subjects

So dimensions are:
    X_i is (n, p)
    Z_i is (n, k)
    W   is (p, k)
    G_i is (k, k)

generate_G generates the N G_i matrices which are positive definite (to avoid degeneracy)
    it takes an optional argument: `diag` to generate diagonal matrices
generate_W generates an orthornormal matrix, which has exactly one non-zero element per row.
    it takes two optional arguments: `ones` to make all its non-zero elements equal to 1,
    and `normalize` to normalize the columns (this happens after `ones` took effect)
generate_Z generates N latent variables from a normal distribution with zero mean and the G_i as covariance.
generate_X generates N observed variables from the latent model above.
generate_all takes all arguments mentioned so far, generates everything, and returns everything.
"""

import numpy as np
from mha.mha import project_W, normalize_columns


def generate_G(N=2, k=5, diag=False):
    G = []
    for i in range(N):
        if diag:
            tmp = np.diag(5 * np.random.rand(k) + 0.1)
        else:
            while 1:
                tmp = np.random.randn(k, k)
                if np.linalg.matrix_rank(tmp) == k:
                    break  # generate an invertible square matrix
            # MM^T is always positive definite for an invertible M
            tmp = tmp.dot(tmp.T)
        G.append(tmp)
    return G


def generate_W(p=50, k=5, normalize=True, ones=False):
    while 1:
        # generate a matrix with positive entries
        W = np.random.rand(p, k)
        # set all but the max entry to zero, per row
        W = project_W(W, ones=ones)
        if np.linalg.matrix_rank(W) == k:
            break  # we want W to be full rank
    if normalize:
        W = normalize_columns(W)
    return W


def generate_Z(n, G):
    Z = []
    k = G[0].shape[0]
    for i in range(len(G)):
        Z.append(np.random.multivariate_normal(
            mean=np.zeros(k), cov=G[i], size=n))
    return Z


def generate_X(n, Z, W):
    N, p = len(Z), W.shape[0]
    E = np.random.randn(N, n, p)
    X = [np.dot(z, W.T) + e for (z, e) in zip(Z, E)]
    return X


def generate_all(n=1000, N=2, p=50, k=5, diag=False, normalize=True, ones=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    G = generate_G(N, k, diag=diag)
    W = generate_W(p, k, normalize=normalize, ones=ones)
    Z = generate_Z(n, G)
    X = generate_X(n, Z, W)
    return X, Z, W, G
