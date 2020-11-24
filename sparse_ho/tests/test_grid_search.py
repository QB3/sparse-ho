import numpy as np
from scipy.sparse import csc_matrix
from sklearn import linear_model
from sparse_ho.utils import Monitor
from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.models import Lasso
from sparse_ho.forward import Forward
from sparse_ho.criterion import HeldOutMSE, SmoothedSURE
from sparse_ho.grid_search import grid_search


n_samples = 100
n_features = 100
n_active = 5
SNR = 3
rho = 0.5

X, y, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=0)
X_s = csc_matrix(X)

idx_train = np.arange(0, 50)
idx_val = np.arange(50, 100)

alpha_max = np.max(np.abs(X[idx_train, :].T @ y[idx_train])) / n_samples
p_alpha = 0.7
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)

log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
tol = 1e-16
max_iter = 1000

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
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max, monitor_grid, max_evals=max_evals,
        tol=1e-5, samp="grid")

    monitor_random = Monitor()
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Forward()
    log_alpha_opt_random, _ = grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max, monitor_random,
        max_evals=max_evals, tol=1e-5, samp="random")

    assert(monitor_random.log_alphas[
        np.argmin(monitor_random.objs)] == log_alpha_opt_random)
    assert(monitor_grid.log_alphas[
        np.argmin(monitor_grid.objs)] == log_alpha_opt_grid)

    monitor_grid = Monitor()
    model = Lasso(estimator=estimator)

    criterion = SmoothedSURE(sigma=sigma_star)
    algo = Forward()
    log_alpha_opt_grid, _ = grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max, monitor_grid, max_evals=max_evals,
        tol=1e-5, samp="grid")

    monitor_random = Monitor()
    criterion = SmoothedSURE(sigma=sigma_star)
    algo = Forward()
    log_alpha_opt_random, _ = grid_search(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max, monitor_random,
        max_evals=max_evals, tol=1e-5, samp="random")

    assert(monitor_random.log_alphas[
        np.argmin(monitor_random.objs)] == log_alpha_opt_random)
    assert(monitor_grid.log_alphas[
        np.argmin(monitor_grid.objs)] == log_alpha_opt_grid)


if __name__ == '__main__':
    test_grid_search()
