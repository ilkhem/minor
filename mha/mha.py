from nilearn import plotting
import numpy as np


class MHA:
    def __init__(self, k, diag=False):
        self.k = k
        self.diag = diag
        self.N = 0
        self.W = None
        self.G = None
        self.n_iters = None

    def __repr__(self):
        mes = "MHA object\n"
        mes += "Number of subjects: " + str(self.N) + "\n"
        mes += "Latent variable dim: " + str(self.k) + "\n"
        if self.diag:
            mes += "Diagonal latent variable covariance"
        else:
            mes += "Full (ie non-diagonal) latent variable covariance"
        return mes

    def fit(self, X, rho=1, tol=0.01, alpha=0.5, c=0.01, max_iter=1000):
        """
        estimate loading matrix and latent variable covariances
        """
        self.N = len(X)
        res = optimize(X, self.k, diag=self.diag,
                       rho=rho, tol=tol, max_iter=max_iter,
                       alpha=alpha, c=c)
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


def project_W(W):
    """a much faster implementation of ProjectMax1 (10+ times faster)"""
    am = W.argmax(axis=1)
    m = W.max(axis=1)
    W = np.zeros_like(W)
    W[np.arange(W.shape[0]), am] = m
    return project_non_negative(W)


def update_A(W, X, diag=False):
    """a much faster implementation of AupdateNonDiag (10+ times faster)
    and uses more than 4 times less memory"""
    tmp = efficient_WTcovW(W, X)
    if not diag:
        tmp = np.eye(W.shape[1]) - np.linalg.pinv(tmp)
    else:
        tmp = np.eye(W.shape[1]) - np.diag(1 / np.diag(tmp))
    # TODO: figure out why he checks for positive eigenvals
    min_ev = np.min(np.linalg.eig(tmp)[0])
    if min_ev <= 0:
        tmp += np.eye(W.shape[1]) * (np.abs(min_ev) + 0.001)
    return tmp


def update_G(W, X, diag=False):
    tmp = efficient_WTcovW(W, X)
    if diag:
        tmp = np.diage(np.diag(tmp))
    return tmp - np.eye(W.shape[1])


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
        # TODO: figure out what stiefiel does
        W_new = W - alpha * grad
        # TODO: is normalizing necessary here?
        # W_new = normalize_columns(project_non_negative(W_new))

        obj = armijo_obj(W, X, A)
        obj_new = armijo_obj(W_new, X, A)

        # the original impl seems to
        # - use W_new - W = - alpha*grad, but then multiplies the whole by another alpha.
        # - compute a central diagof things, which I don't understand
        # - add a +0.001 which is also mysterious
        # TODO: figure out why he does that
        if obj_new <= obj - c * alpha * np.linalg.norm(grad) ** 2:
            break
        alpha *= tau
        i += 1
        if i > max_iter:
            break
    return W_new


def init_W(X, k):
    p = X[0].shape[1]
    mean_cov = np.zeros((p, p))  # mean covariance across all subjects
    for i in range(len(X)):
        mean_cov += (1.0 / len(X)) * np.cov(X[i], rowvar=False)
    eig_vals, eig_vecs = np.linalg.eig(mean_cov)
    idx = eig_vals.argsort()[::-1][:k]
    W = eig_vecs[:, idx]
    # TODO: is the sign flip necessary?
    return project_non_negative(W * (2 * (W.sum(0) >= 0) - 1))


def optimize(X, k, diag=False, rho=1, tol=0.01, alpha=0.5, c=0.01, max_iter=1000):
    N = len(X)
    # define initial parameters:
    Lambda = np.zeros((k, k))
    W = init_W(X, k)
    W = normalize_columns(W)
    W_old = np.copy(W)

    for n_iters in range(max_iter):
        # -------- update W matrix --------
        # first compute A
        A = [update_A(W, X[i], diag=diag) for i in range(N)]
        AA = [0.5 * a.dot(a) - a for a in A]
        # compute gradient of SM objective with respect to W
        grad = np.zeros(W.shape)
        for i in range(N):
            # TODO: why divide by N?
            grad += efficient_gradJ(W, X[i], AA[i]) / N
        grad += rho * (W.dot(W.T.dot(W) - np.eye(k) + Lambda / rho))
        # compute armijo update:
        W = update_W(W, grad, AA, X, alpha=alpha, c=c)
        W = normalize_columns(project_non_negative(W))  # to ensure non-negativity

        # -------- update Lagrange multipler --------
        Lambda = Lambda + rho * (W.T.dot(W) - np.eye(k))

        # -------- check for convergence --------
        if np.linalg.norm(W - W_old) < tol:
            break
        else:
            W_old = np.copy(W)

    W = normalize_columns(project_W(W))
    G = [update_G(W, X[i], diag=diag) for i in range(N)]

    return {"W": W, "G": G, "n_iters": n_iters}
