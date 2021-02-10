from MHA_backend import optimize
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
