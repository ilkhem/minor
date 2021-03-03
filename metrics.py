import numpy as np
from mha.mha import project_W
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance as spd

def cluster_score(W, W_true, distance='jaccard'):
    """
    Permute the columns of the estimate W in order to be aligned with the columns of W_true.
    This is because the model is invariant to permutations of the columns of W.
    We do this by solving a linear sum assignment problem using the Hungarian algorithm.

    Parameters:
    ----------
    W: np.ndarray
        Estimated loading matrix
    W_true: np.ndarray
        True loading matrix
    distance: str
        Distance to compute between the true and the estimated clusters.
        Can be either `jaccard`, `hamming` or `kulsinski`.

    Returns:
    ----------
    score: float
        The optimal assignment cost.
    W_aligned: np.ndarray
        A copy of W with its columns rearranged according to the optimal alignment.
    dist: float
        The distance between the true and the estimated clusters.
    alignment: tuple of (np.ndarray, np.ndarray)
        The row-idx and column-idx of the optinal alignment
    """
    # process distance argument
    assert distance in ['jaccard', 'hamming', 'kulsinski']
    # make sure W and W_true are {0, 1}-values
    W = project_W(W, ones=True)
    W_true = project_W(W_true, ones=True)
    # compute cost matrix
    cost = np.sum((W[:, :, None] - W_true[:, None, :]) ** 2, axis=0)
    # solve the assignment problem
    alignment = linear_sum_assignment(cost)
    # compute score
    score = np.mean(cost[alignment])
    # create an aligned version of W
    W_aligned = W.copy()[:, alignment[1]]
    # compare clusters
    am_W, am_W_true = W_aligned.argmax(1), W_true.argmax(1)
    if distance == 'jaccard':
        dist = spd.jaccard(am_W, am_W_true)
    elif distance == 'hamming':
        dist = spd.hamming(am_W, am_W_true)
    else:
        dist = spd.kulsinski(am_W, am_W_true)
    return score, W_aligned, dist, alignment

