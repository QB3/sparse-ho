import numpy as np
# from scipy.sparse import csc_matrix
from sklearn import linear_model
from celer.datasets import make_correlated_data

from sparse_ho.utils import Monitor
from sparse_ho.models import Lasso
from sparse_ho import Forward
from sparse_ho.criterion import HeldOutMSE, FiniteDiffMonteCarloSure
from sparse_ho.grid_search import grid_search


n_samples = 100
n_features = 100
snr = 3
rho = 0.5

X, y, _ = make_correlated_data(
    n_samples, n_features, rho=rho, snr=snr, random_state=42)
# XXX TODO add test for sparse matrices
# X_s = csc_matrix(X)
sigma_star = 0.1

idx_train = np.arange(0, 50)
idx_val = np.arange(50, 100)

alpha_max = np.max(np.abs(X[idx_train, :].T @ y[idx_train])) / len(idx_train)

log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))

log_alpha_max = np.log(alpha_max)
log_alpha_min = np.log(0.0001 * alpha_max)

estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)


def test_grid_search():
    max_evals = 5

    monitor_grid = Monitor()
    model = Lasso(estimator=estimator)
    criterion = HeldOutMSE(idx_train, idx_train)
    algo = Forward()
    log_alpha_opt_grid, _ = grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max,
        monitor_grid, max_evals=max_evals,
        tol=1e-5, samp="grid")

    monitor_random = Monitor()
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Forward()
    log_alpha_opt_random, _ = grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max,
        monitor_random,
        max_evals=max_evals, tol=1e-5, samp="random")

    assert(monitor_random.log_alphas[
        np.argmin(monitor_random.objs)] == log_alpha_opt_random)
    assert(monitor_grid.log_alphas[
        np.argmin(monitor_grid.objs)] == log_alpha_opt_grid)

    monitor_grid = Monitor()
    model = Lasso(estimator=estimator)

    criterion = FiniteDiffMonteCarloSure(sigma=sigma_star)
    algo = Forward()
    log_alpha_opt_grid, _ = grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max,
        monitor_grid, max_evals=max_evals,
        tol=1e-5, samp="grid")

    monitor_random = Monitor()
    criterion = FiniteDiffMonteCarloSure(sigma=sigma_star)
    algo = Forward()
    log_alpha_opt_random, _ = grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max,
        monitor_random,
        max_evals=max_evals, tol=1e-5, samp="random")

    assert(monitor_random.log_alphas[
        np.argmin(monitor_random.objs)] == log_alpha_opt_random)
    assert(monitor_grid.log_alphas[
        np.argmin(monitor_grid.objs)] == log_alpha_opt_grid)


if __name__ == '__main__':
    test_grid_search()
