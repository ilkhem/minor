import numpy as np
from mha.mha import project_W
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as spd


def get_alignment(W, W_true, cost_dist='euclidean'):
    """
    Permutes the columns of the estimate W in order to be aligned with the columns of W_true.
    This is because the model is invariant to permutations of the columns of W.
    We do this by solving a linear sum assignment problem using the Hungarian algorithm, based
    on a specifed cost function.
    """
    # compute cost matrix
    # C_ij = cost to match i-th col of W_true with j-th col of W
    if cost_dist == 'euclidean':
        cost = np.sum((W_true[:, :, None] - W[:, None, :]) ** 2, axis=0)
    else:
        k = W.shape[1]
        WW = project_W(W, ones=True)
        Wt = project_W(W_true, ones=True)
        cost = np.zeros((k, k))
        for i in range(k):
            for j in range(k):
                cost[i, j] = spd.hamming(WW[:, j], Wt[:, i])
    # solve the assignment problem
    alignment = linear_sum_assignment(cost)
    # compute score
    score = np.mean(cost[alignment])
    return alignment[1], score


def align_from_permutation(W, G, alignment=None):
    if alignment is None:
        alignment = np.arange(W.shape[1])
    inverse_alignment = np.argsort(alignment)
    Wal = W.copy()[:, alignment]
    Gal = [Gi.copy()[inverse_alignment][:, alignment] for Gi in G]
    return Wal, Gal


def align(W, W_true, G, cost_dist='euclidean'):
    alignment, score = get_alignment(W, W_true, cost_dist)
    Wal, Gal = align_from_permutation(W, G, alignment)
    return Wal, Gal, alignment, score


def cluster_score(W, W_true, cost_dist='euclidean'):
    """
    Subsequently, computes several distances between the true and estimated loading matrices.

    Parameters:
    ----------
    W: np.ndarray
        Estimated loading matrix
    W_true: np.ndarray
        True loading matrix

    Returns:
    ----------
    distances: dict
        The distance between the true and the estimated clusters.
        Computes the "Jaccard", "Hamming" and "Kulsinski" distances and return them as a dict.
    score: float
        The optimal assignment cost.
    W_aligned: np.ndarray
        A copy of W with its columns rearranged according to the optimal alignment.
    alignment: tuple of (np.ndarray, np.ndarray)
        The row-idx and column-idx of the optimal alignment.
    """
    # align W to W_true by shuffling its columns
    alignment, score = get_alignment(W, W_true, cost_dist)
    # create an aligned version of W by permuting its columns
    W_aligned = W.copy()[:, alignment[1]]
    # compare clusters
    am_W, am_W_true = W_aligned.argmax(1), W_true.argmax(1)
    distances = {}
    distances['jaccard'] = spd.jaccard(am_W, am_W_true)
    distances['hamming'] = spd.hamming(am_W, am_W_true)
    distances['kulsinski'] = spd.kulsinski(am_W, am_W_true)
    return distances, score, W_aligned, alignment


def single_covariance_mse(Gi, Gi_true, alignment=None):
    if alignment is None:
        alignment = np.arange(Gi.shape[0])
    inverse_alignment = np.argsort(alignment)
    return np.mean((Gi[inverse_alignment][:, alignment] - Gi_true) ** 2)


def covariance_mse(G, G_true, alignment=None):
    return [single_covariance_mse(G[i], G_true[i], alignment) for i in range(len(G))]


def distances(W, W_true, G, G_true, alignment=None):
    if alignment is None:
        alignment = np.arange(W.shape[1])
    Wal, Gal = align_from_permutation(W, G, alignment)
    am_W, am_W_true = Wal.argmax(1), W_true.argmax(1)
    distances = {}
    distances['jaccard'] = spd.jaccard(am_W, am_W_true)
    distances['hamming'] = spd.hamming(am_W, am_W_true)
    distances['kulsinski'] = spd.kulsinski(am_W, am_W_true)
    cov_mse = covariance_mse(Gal, G_true)
    distances['cov_mse'] = cov_mse
    distances['cov_mse_mean'] = np.mean(cov_mse)
    distances['cov_mse_max'] = np.max(cov_mse)
    return distances


def log_likelihood(X, W, G):
    """
    Compute the likelihood of an N-sample X, which follows an MHA model,
    i.e. X has a Gaussian distribution with 0 mean and covariance Sigma,
    where Sigma = WGW^T + I, W is (p, k) and G is (k, k).
    When p >> k, computing Sigma and plugging it into the formula for the ll
    is very costly, and total cost is O(p^3).
    The below implementation uses the fact that Sigma can be written as a function
    of the lower dimensional matrices G and W, and reduces the cost to O(max(p^2, k^3)).
    """
    p, k = W.shape
    # compute log(det(Sigma)) efficiently, using Sylvester's determinant theorem
    WtW = W.T.dot(W)
    elds = np.log(np.linalg.det(np.eye(k) + WtW.dot(G)))
    # compute x^T.Sigma^-1.x efficiently, using Woodbury matrix lemma + some algebra
    WtX = X.dot(W)
    GG = G.dot(np.linalg.inv(WtW.dot(G) + np.eye(k)))
    etxsx = np.einsum('bi,bi->b', X, X) - \
        np.einsum('bi,ij,bj->b', WtX, GG, WtX)
    ll = - .5 * (np.log(2 * np.pi) * p + elds + etxsx)
    return ll
