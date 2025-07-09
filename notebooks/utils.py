import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from scipy.linalg import lstsq
from scipy.spatial.distance import pdist

from tikreg.utils import explainable_variance
from himalaya.scoring import correlation_score

from collections import defaultdict


def make_correlated_signal(Y, rho, rng=None):
    """Return correlated signal with a desired correlation `rho`.

    Parameters
    ----------
    Y : array (n_samples, n_channels)
        input signal
    rho : float or array (n_channels)
        desired correlation(s)

    Returns
    -------
    Z : array (n_samples, n_channels)
        correlated signal with input signal, such that
        corr(Y, Z) = rho for each channel
    """
    if rng is None:
        rng = np.random.RandomState()
    # method from https://stats.stackexchange.com/a/313138
    n_samples, n_channels = Y.shape
    if isinstance(rho, float):
        rho = [rho] * n_channels
    Z = np.zeros_like(Y)
    # zscore Y to get exact correlation scores
    Y = zscore(Y, 0)
    for i in range(n_channels):
        y = Y[:, i]
        r = rho[i]
        # handle case rho == ± 1
        if np.allclose(np.abs(r), 1):
            Z[:, i] = np.sign(r) * y
        else:
            # make a random vector to get projection
            X = rng.randn(n_samples, 1)
            # find orthogonal vector
            y = y[:, None]
            B, *_ = lstsq(y, X)
            E = X - y.dot(B)
            # build correlated vector
            z = r * E.std(0) * y + np.sqrt(1 - r**2) * y.std(0) * E
            Z[:, i] = z.squeeze()
    return Z


def test_make_correlated_signal():
    # w/ more samples the correlation gets closer to the desired value
    n_samples = 10000
    n_channels = 10
    Y = np.random.randn(n_samples, n_channels)
    # get random correlation values
    rhos = np.random.uniform(size=n_channels)
    # make correlated signal
    Z = make_correlated_signal(Y, rhos)
    # check that we are close enough
    # assert np.all(np.abs(correlation_score(Y, Z) - rhos) < 1e-3)
    assert np.allclose(correlation_score(Y, Z), rhos)


test_make_correlated_signal()


def scatter(x, y, lims=None, ax=None, mask_nans=True, color="k", alpha=0.5):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    if mask_nans:
        mask = (~np.isnan(x)) & (~np.isnan(y))
        x = x[mask]
        y = y[mask]
    ax.scatter(x, y, alpha=alpha, marker=".", color=color)
    ax.axis("square")

    if lims is None:
        lims = [np.min([x, y]) - 0.05, np.max([x, y]) + 0.05]
    ax.plot(lims, lims, color="lightgray", zorder=0)
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid()
    ax.set_axisbelow(True)
    return ax


def compute_ccnorm(Yhat, Yrep, dozscore=False):
    if dozscore:
        Yrep = zscore(Yrep, 1)
        # XXX: note this is wrong! this is the reason why dividing by EV
        # causes overshoots. we are discarding the variance of the average over repeats
        # and this is bad!
        Y = zscore(Yrep.mean(0), 0)
        Yhat = zscore(Yhat, 0)
    else:
        Y = Yrep.mean(0)
    N, n_samples, _ = Yrep.shape
    # precompute
    VarY = Y.var(0)
    VarYhat = Yhat.var(0)
    CovYYh = ((Y - Y.mean(0)) * (Yhat - Yhat.mean(0))).sum(0) / (n_samples - 1)
    SP = (Yrep.sum(0).var(0) - Yrep.var(1).sum(0)) / (N * (N - 1))
    # if SP < 0, set to 0
    mask_SP = SP < 0
    SP[mask_SP] = 0.0
    CCnorm = CovYYh / np.sqrt(VarYhat * SP)
    # CCnorm is not defined for SP < 0
    CCnorm[mask_SP] = np.nan
    CCmax = np.sqrt(SP / VarY)
    CCabs = CovYYh / np.sqrt(VarY * VarYhat)
    return CCnorm, CCabs, CCmax, SP
