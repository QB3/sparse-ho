"""Utility files for the MEG data examples."""

import numpy as np
from mne.inverse_sparse.mxne_inverse import (
    _prepare_gain, is_fixed_orient, _reapply_source_weighting,
    _make_sparse_stc)
from celer import Lasso as celer_Lasso
from sparse_ho.utils import Monitor
from sparse_ho.models import WeightedLasso, Lasso
from sparse_ho.criterion import FiniteDiffMonteCarloSure
from sparse_ho import Implicit
from sparse_ho.ho import grad_search
from sparse_ho.optimizers import GradientDescent


def apply_solver(
        evoked, forward, noise_cov, loose=0.2, depth=0.8, p_alpha0=0.7,
        model_name="wlasso"):
    """Call a custom solver on evoked data.

    This function does all the necessary computation:

    - to select the channels in the forward given the available ones in
      the data
    - to take into account the noise covariance and do the spatial whitening
    - to apply loose orientation constraint as MNE solvers
    - to apply a weighting of the columns of the forward operator as in the
      weighted Minimum Norm formulation in order to limit the problem
      of depth bias.


    Parameters
    ----------
    XXXXXXXXXXXXXXXX Solver is not a parameter of the function, TODO
    solver : callable
        The solver takes 3 parameters: data M, gain matrix G, number of
        dipoles orientations per location (1 or 3). A solver shall return
        2 variables: X which contains the time series of the active dipoles
        and an active set which is a boolean mask to specify what dipoles are
        present in X.
    evoked : instance of mne.Evoked
        The evoked data
    forward : instance of Forward
        The forward solution.
    noise_cov : instance of Covariance
        The noise covariance.
    loose : float in [0, 1] | 'auto'
        Value that weights the source variances of the dipole components
        that are parallel (tangential) to the cortical surface. If loose
        is 0 then the solution is computed with fixed orientation.
        If loose is 1, it corresponds to free orientations.
        The default value ('auto') is set to 0.2 for surface-oriented source
        space and set to 1.0 for volumic or discrete source space.
    depth : None | float in [0, 1]
        Depth weighting coefficients. If None, no depth weighting is performed.

    Returns
    -------
    stc : instance of SourceEstimate
        The source estimates.
    """
    all_ch_names = evoked.ch_names

    # Handle depth weighting and whitening (here is no weights)
    forward, gain, gain_info, whitener, source_weighting, mask = _prepare_gain(
        forward, evoked.info, noise_cov, pca=False, depth=depth,
        loose=0, weights=None, weights_min=None, rank=None)

    # Select channels of interest
    sel = [all_ch_names.index(name) for name in gain_info['ch_names']]
    M = evoked.data[sel]

    # Whiten data
    M = np.dot(whitener, M)

    n_orient = 1 if is_fixed_orient(forward) else 3

    X, active_set, monitor = solver(
        M, gain, n_orient, evoked.nave, p_alpha0=p_alpha0,
        model_name=model_name)
    X = _reapply_source_weighting(X, source_weighting, active_set)

    stc = _make_sparse_stc(X, active_set, forward, tmin=evoked.times[0],
                           tstep=1. / evoked.info['sfreq'])

    return stc, monitor


###############################################################################
# Define your solver

def solver(
        y_train, X_train, n_orient, nave, p_alpha0=0.7, model_name="wlasso"):
    n_times = y_train.shape[1]
    idx_max = np.argmax(np.sum(y_train ** 2, axis=0))
    y_train = y_train[:, idx_max]

    n_samples, n_features = X_train.shape
    alpha_max_old = (np.abs(X_train.T @ y_train)).max() / n_samples
    X_train /= alpha_max_old

    alpha_max = (np.abs(X_train.T @ y_train)).max() / n_samples
    alpha0 = p_alpha0 * alpha_max

    estimator = celer_Lasso(
        fit_intercept=False, max_iter=100, warm_start=True,
        tol=1e-3)
    if model_name == "wlasso":
        alpha0 = alpha0 * np.ones(n_features)
        model = WeightedLasso(estimator=estimator)

    else:
        model = Lasso(estimator=estimator)

    sigma = 1 / np.sqrt(nave)
    criterion = FiniteDiffMonteCarloSure(sigma=sigma)
    algo = Implicit()
    optimizer = GradientDescent(
        n_outer=4, tol=1e-7, verbose=True, p_grad_norm=1.9)
    monitor = Monitor()
    grad_search(algo, criterion, model, optimizer,
                X_train, y_train, alpha0, monitor)

    X = criterion.dense0[:, np.newaxis] * np.ones((1, n_times))
    active_set = criterion.mask0
    X /= alpha_max_old

    return X, active_set, monitor
