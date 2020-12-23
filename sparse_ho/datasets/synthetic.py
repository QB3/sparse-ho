"""File to generate synthetic data
"""


import numpy as np

from numpy.linalg import norm
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state


def get_synt_data(n_samples=20, n_features=20, n_times=30, n_active=3,
                  rho=0.3, snr=1./7, seed=0):
    """Simulate artificial multitask linear regression data, with correlation
    between features i and j equal to rho ** |i - j|

    Parameters:
    -----------
    n_samples: int
        Number of samples (channels in MEG).
    n_features: int
        Number of features (candidate sources in MEG).
    n_times: int
        Number of tasks (time points in MEG).
    n_active: int
        True support size (number of active sources in MEG).
    rho: float
        Coefficient of correlation for the Toeplitz-correlated design matrix.
    snr: float
        Signal to noise ratio.
    seed: int

    Returns
    -------
    X: np.array, shape (n_samples, n_features)
        Design matrix (forward operator in MEG).
    Y: np.array, shape (n_samples, n_times)
        Observations.
    B_star: np.array, shape (n_features, n_times)
        True regression coefficients.
    """
    rng = check_random_state(seed)
    if rho > 0:
        vect = rho ** np.arange(n_features)
        covar = toeplitz(vect, vect)
        X = rng.multivariate_normal(np.zeros(n_features), covar, n_samples)
    else:
        X = rng.randn(n_samples, n_features)

    # creates the signal XB
    B_star = np.zeros([n_features, n_times])
    supp = rng.choice(n_features, n_active, replace=False)
    B_star[supp, :] = 1

    Y = X @ B_star
    noise = rng.randn(n_samples, n_times)
    sigma_star = norm(Y, ord='fro') / (norm(noise, ord='fro') * snr)
    noise *= sigma_star
    Y += noise

    if n_times == 1:
        return X, Y[:, 0], B_star[:, 0], noise, sigma_star
    else:
        return X, Y, B_star, noise, sigma_star
