from MHA_backend import nonNegativeCovFactor_LagrangeMult
from nilearn import plotting
import numpy as np


class MHA:
    """
    class for MHA object

    INPUT:
            -
    """

    def __init__(self, Shat, k, diagG=False):
        self.Shat = Shat
        self.k = k
        self.diagG = diagG
        self.W = None
        self.G = None
        self.iter = None

    def __repr__(self):
        mes = "MHA object\n"
        mes += "Number of subjects: " + str(len(self.Shat)) + "\n"
        mes += "Latent variable dim: " + str(self.k) + "\n"
        if self.diagG:
            mes += "Diagonal latent variable covariance"
        else:
            mes += "Full (ie non-diagonal) latent variable covariance"
        return mes

    def fit(self, lagParam=1, tol=0.01, alphaArmijo=0.5, maxIter=1000):
        """
        estimate loading matrix and latent variable covariances
        """
        res = nonNegativeCovFactor_LagrangeMult(
            Shat=self.Shat,
            k=self.k,
            diagG=self.diagG,
            lagParam=lagParam,
            tol=tol,
            alphaArmijo=alphaArmijo,
            maxIter=maxIter,
        )
        self.W = res["W"]
        self.G = res["G"]
        self.iter = res["iter"]

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
