import pytest
import numpy as np
from scipy.sparse import csc_matrix
from sklearn import linear_model
from sklearn.model_selection import KFold
import celer
from celer.datasets import make_correlated_data

from sparse_ho.utils import Monitor
from sparse_ho.models import Lasso
from sparse_ho import Forward
from sparse_ho.criterion import (
    HeldOutMSE, FiniteDiffMonteCarloSure, CrossVal, HeldOutLogistic)
from sparse_ho.grid_search import grid_search


n_samples = 100
n_features = 100
snr = 3
corr = 0.5

X, y, _ = make_correlated_data(
    n_samples, n_features, corr=corr, snr=snr, random_state=42)
sigma_star = 0.1
y = np.sign(y)
X_s = csc_matrix(X)

idx_train = np.arange(0, 50)
idx_val = np.arange(50, 100)

alpha_max = np.max(np.abs(X[idx_train, :].T @ y[idx_train])) / len(idx_train)

log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
log_alpha_max = np.log(alpha_max)
log_alpha_min = np.log(0.0001 * alpha_max)

estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=10000, warm_start=True)
model = Lasso(estimator=estimator)

tol = 1e-8

# Set models to be tested
models = {}
models["lasso"] = Lasso(estimator=None)

models["lasso_custom"] = Lasso(estimator=celer.Lasso(
    warm_start=True, fit_intercept=False))


@pytest.mark.parametrize('model_name', list(models.keys()))
@pytest.mark.parametrize('XX', [X, X_s])
def test_cross_val_criterion(model_name, XX):
    model = models[model_name]
    alpha_min = alpha_max / 10
    log_alpha_max = np.log(alpha_max)
    log_alpha_min = np.log(alpha_min)
    max_iter = 10000
    n_alphas = 10
    kf = KFold(n_splits=5, shuffle=True, random_state=56)

    monitor_grid = Monitor()
    if model_name.startswith("lasso"):
        sub_crit = HeldOutMSE(None, None)
    else:
        sub_crit = HeldOutLogistic(None, None)
    criterion = CrossVal(sub_crit, cv=kf)
    algo = Forward()
    grid_search(
        algo, criterion, model, XX, y, log_alpha_min, log_alpha_max,
        monitor_grid, max_evals=n_alphas, tol=tol)

    if model_name.startswith("lasso"):
        reg = linear_model.LassoCV(
            cv=kf, verbose=True, tol=tol, fit_intercept=False,
            alphas=np.geomspace(alpha_max, alpha_min, num=n_alphas),
            max_iter=max_iter).fit(X, y)
    else:
        reg = linear_model.LogisticRegressionCV(
            cv=kf, verbose=True, tol=tol, fit_intercept=False,
            Cs=len(idx_train) / np.geomspace(
                alpha_max, alpha_min, num=n_alphas),
            max_iter=max_iter, penalty='l1', solver='liblinear').fit(X, y)
    reg.score(XX, y)
    if model_name.startswith("lasso"):
        objs_grid_sk = reg.mse_path_.mean(axis=1)
    else:
        objs_grid_sk = reg.scores_[1.0].mean(axis=1)
    # these 2 value should be the same
    (objs_grid_sk - np.array(monitor_grid.objs))
    np.testing.assert_allclose(objs_grid_sk, monitor_grid.objs)


# TOD0 factorize this tests
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

    np.testing.assert_allclose(monitor_random.log_alphas[
        np.argmin(monitor_random.objs)], log_alpha_opt_random)
    np.testing.assert_allclose(monitor_grid.log_alphas[
        np.argmin(monitor_grid.objs)], log_alpha_opt_grid)

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

    np.testing.assert_allclose(monitor_random.log_alphas[
        np.argmin(monitor_random.objs)], log_alpha_opt_random)
    np.testing.assert_allclose(monitor_grid.log_alphas[
        np.argmin(monitor_grid.objs)], log_alpha_opt_grid)


if __name__ == '__main__':
    for model_name in models.keys():
        test_cross_val_criterion(model_name)
    # test_grid_search()
