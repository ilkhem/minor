import numpy as np
from mha.mha import project_W
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as spd


def cluster_score(W, W_true, cost_dist='euclidean'):
    """
    Permutes the columns of the estimate W in order to be aligned with the columns of W_true.
    This is because the model is invariant to permutations of the columns of W.
    We do this by solving a linear sum assignment problem using the Hungarian algorithm, based
    on a specifed cost function.

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
    # compute cost matrix
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
    # create an aligned version of W
    W_aligned = W.copy()[:, alignment[1]]
    # compare clusters
    am_W, am_W_true = W_aligned.argmax(1), W_true.argmax(1)
    distances = {}
    distances['jaccard'] = spd.jaccard(am_W, am_W_true)
    distances['hamming'] = spd.hamming(am_W, am_W_true)
    distances['kulsinski'] = spd.kulsinski(am_W, am_W_true)
    return distances, score, W_aligned, alignment


def covariance_mse(Gi, Gi_true):
    return np.mean((Gi - Gi_true) ** 2)


def nll_unseen_data(Gi, X):
    return .5 * np.log(np.det(Gi)) + .5 * np.trace(np.cov(X, rowvar=False).dot(np.linalg.inv(Gi)))
