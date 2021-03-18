from nilearn import plotting
import numpy as np
from tqdm import tqdm
from sklearn.utils.extmath import randomized_svd, svd_flip
from scipy.sparse.linalg import svds
from scipy.linalg import svd


class MHA:
    def __init__(self, k, diag=False, verbose=False, init_method='random_svd'):
        self.k = k
        self.diag = diag
        self.N = 0
        self.W = None
        self.G = None
        self.n_iters = None
        self.verbose = verbose
        self.init_method = init_method

    def __repr__(self):
        mes = "MHA object\n"
        mes += "Number of subjects: " + str(self.N) + "\n"
        mes += "Latent variable dim: " + str(self.k) + "\n"
        if self.diag:
            mes += "Diagonal latent variable covariance"
        else:
            mes += "Full (ie non-diagonal) latent variable covariance"
        return mes

    def fit(self, X, rho=1, tol=0.01, alpha=0.5, c=0.01, max_iter=10000):
        """
        estimate loading matrix and latent variable covariances
        """
        self.N = len(X)
        res = optimize(X, self.k, diag=self.diag,
                       rho=rho, tol=tol, max_iter=max_iter,
                       alpha=alpha, c=c,
                       verbose=self.verbose,
                       init_method=self.init_method)
        self.W = res["W"]
        self.G = res["G"]
        self.n_iters = res["n_iters"]

    def transform(self, Xnew):
        """
        apply projection matrix, W, to new data

        INPUT:
                - Xnew: list of numpy array, each entry should be an n by p array of n observations for p random variables
        """
        ProjXnew = [X.dot(self.W) for X in Xnew]
        return ProjXnew

    def plot(self, ROIcoord, clusterID, title):
        """
        INPUT:
                - ROIcoord: MNNI coordinates
                - clusterID: which cluster should we plot

        """
        ii = np.where(self.W[:, clusterID] != 0)[0]
        RandomMat = np.cov(
            np.random.random((10, len(ii))).T
        )  # this is just a place holder, we will not plot any of it!

        # we just plot the result
        plotting.plot_connectome(
            RandomMat,
            ROIcoord[ii, :],
            node_color="black",
            annotate=False,
            display_mode="ortho",
            edge_kwargs={"alpha": 0},
            node_size=50,
            title=title,
        )


def project_non_negative(W):
    return W * (W > 0)


def normalize_columns(W):
    """a faster implementation of normalizeColumns (~3 times faster)"""
    return W / (np.linalg.norm(W, axis=0)[None, :] + 0.001)


def efficient_WTcovW(W, X):
    """a fast and memory efficient implementation of W.T.dot(cov(X)).dot(W)"""
    tmp = (X - X.mean(0)).dot(W) / np.sqrt(X.shape[0] - 1)
    return tmp.T.dot(tmp)


def efficient_gradJ(W, X, AA):
    """a fast and memory efficient implementation of cov(X).dot(W).dot(AA)"""
    X_tilde = (X - X.mean(0)) / np.sqrt(X.shape[0] - 1)
    tmp = X_tilde.dot(W)
    tmp = tmp.dot(AA)
    return X_tilde.T.dot(tmp)


def project_W(W, ones=False):
    """a much faster implementation of ProjectMax1 (10+ times faster)"""
    am = W.argmax(axis=1)
    m = W.max(axis=1) if not ones else 1
    W = np.zeros_like(W)
    W[np.arange(W.shape[0]), am] = m
    return W


def update_A(W, X, diag=False):
    """a much faster implementation of AupdateNonDiag (10+ times faster)
    and uses more than 4 times less memory"""
    A = efficient_WTcovW(W, X)
    if not diag:
        A = np.eye(W.shape[1], dtype=X.dtype) - np.linalg.pinv(A)
    else:
        A = np.eye(W.shape[1], dtype=X.dtype) - np.diag(1 / np.diag(A))
    # the operation above sometimes results in a non pos-def matrix, we
    # take care of this by ensuring that the smallest eig-val of A is pos.
    min_ev = np.min(np.linalg.eig(A)[0])
    if min_ev <= 0:
        A += np.eye(W.shape[1], dtype=X.dtype) * (np.abs(min_ev) + 0.001)
    return A


def update_G(W, X, diag=False):
    tmp = efficient_WTcovW(W, X)
    if diag:
        tmp = np.diag(np.diag(tmp))
    return tmp - np.eye(W.shape[1], dtype=X.dtype)


def armijo_obj(W, X, A):
    # TODO: what is this obj???
    # expects the following dims:
    # W: (p, k)
    # X: list-like of N (n_i, p) matrices
    # A: list-like of N (k, k) matrices
    obj = 0
    for x, a in zip(X, A):
        tmp = efficient_WTcovW(W, x)
        obj += np.trace(tmp.dot(a))
    return obj


def update_W(W, grad, A, X, alpha=0.5, c=0.001, tau=0.5, max_iter=1000):
    grad = normalize_columns(grad)
    i = 1
    while True:
        W_new = W - alpha * grad
        # W_new = normalize_columns(project_non_negative(W_new))
        obj = armijo_obj(W, X, A)
        obj_new = armijo_obj(W_new, X, A)
        if obj_new <= obj - c * alpha * np.linalg.norm(grad) ** 2:
            break
        alpha *= tau
        i += 1
        if i > max_iter:
            break
    return W_new


def init_W(X, k, method='random_svd', project=False, verbose=False, svd_kwargs={}):
    """
    Initialize W, either from data X through a PCA, or randomly.

    Parameters:
    ----------
    X: list of np.ndarray
        List containing an observation array per subject.
    k: int
        Number of components.
    method: string
        Initialization method. Can be one of the following:
            - random_svd: k-first principal components, computed using the `skleanr.utils.extmath.randomized_svd`
            method on the concatenation of all observations in X. This method is very fast and memory efficient,
            but only provides an approximation of the PCs.
            - sparse_svd: k-first principal components, computed using the `scipy.sparse.linalg.svds` method
            on the concatenation of all observations in X. This method is exact and memory efficient, as it only
            computes the first k PCs, but is somewhat slow,
            - truncated_svd: k-first principal components, computed using the `scip.linalg.svd` method
            on the concatenation of all observations in X. This method is exact, somewhat fast, but is not
            memory efficient, as it computes all PCs, and performs eh truncation at the end.
            - random: Randomly initialize an orthonormal matrix with positive entries. This method is very fast
            and memory efficient, but results in a mediocre initialization.
        Defaults to `random_svd`.
    project: boolean
        Project resulting W onto the non-negative quadrant.
    verbose: boolean
        Toggle verbosity.
    svd_kwargs: dict
        Optional keyword arguments for the SVD algorithms.

    Returns:
    ----------
    W: np.ndarray
        An initialization of the loading matrix W.
    """
    if verbose:
        print("Initializing W ...")
    stacked_X = np.concatenate(X)
    if method == 'random_svd':
        _, _, W = randomized_svd(
            stacked_X, n_components=k, flip_sign=True, **svd_kwargs)
    elif method == 'sparse_svd':
        u, _, W = svds(stacked_X, k=k, which='LM', **svd_kwargs)
        u, W = svd_flip(u[:, ::-1], W[::-1])
    elif method == 'truncated_svd':
        u, _, W = svd(stacked_X, full_matrices=False, **svd_kwargs)
        u, W = svd_flip(u, W)
        W = W[:k]
    elif method == 'random':
        p = X[0].shape[1]
        while 1:
            # generate a matrix with positive entries
            W = np.random.rand(p, k)
            # set all but the max entry to zero, per row
            W = project_W(W, ones=True)
            if np.linalg.matrix_rank(W) == k:
                break  # we want W to be full rank
        W = normalize_columns(W).T
    else:
        raise ValueError(f'Unknown method {method}')
    W = W.T
    W = (W * (2 * (W.sum(0) >= 0) - 1)).astype(X[0].dtype)
    return project_non_negative(W) if project else W


def optimize(X, k, diag=False, rho=1, tol=0.01, alpha=0.5, c=0.01, max_iter=1000,
             init_method='random_svd', svd_kwargs={}, verbose=False):
    N = len(X)
    dt = X[0].dtype
    # define initial parameters:
    Lambda = np.zeros((k, k), dtype=dt)
    W = init_W(X, k, method=init_method,
               verbose=verbose, svd_kwargs=svd_kwargs)
    W = normalize_columns(W)
    W_old = np.copy(W)

    iterator = range(max_iter)
    if verbose:
        print("Optimization starting ...")
        iterator = tqdm(iterator, desc="Optimization:")
    for n_iters in iterator:
        # -------- update W matrix --------
        # first compute A
        A = [update_A(W, X[i], diag=diag) for i in range(N)]
        AA = [0.5 * a.dot(a) - a for a in A]
        # compute gradient of SM objective with respect to W
        grad = np.zeros(W.shape, dtype=dt)
        for i in range(N):
            # grad += efficient_gradJ(W, X[i], AA[i]) / float(N)
            grad += efficient_gradJ(W, X[i], AA[i])
        grad += rho * (W.dot(W.T.dot(W) - np.eye(k, dtype=dt) + Lambda / rho))
        # compute armijo update:
        W = update_W(W, grad, AA, X, alpha=alpha, c=c)
        # to ensure non-negativity
        W = normalize_columns(project_non_negative(W))

        # -------- update Lagrange multipler --------
        Lambda = Lambda + rho * (W.T.dot(W) - np.eye(k, dtype=dt))

        # -------- check for convergence --------
        if np.linalg.norm(W - W_old) < tol:
            break
        else:
            W_old = np.copy(W)

    W = normalize_columns(project_W(W))
    G = [update_G(W, X[i], diag=diag) for i in range(N)]

    return {"W": W, "G": G, "n_iters": n_iters}
