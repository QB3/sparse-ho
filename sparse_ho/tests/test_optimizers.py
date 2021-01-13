import numpy as np
from scipy.sparse import csc_matrix
import pytest
from sklearn import linear_model

from sparse_ho.utils import Monitor
from sparse_ho.datasets import get_synt_data
from sparse_ho.models import Lasso

from sparse_ho import Forward
from sparse_ho import ImplicitForward
from sparse_ho import Implicit
from sparse_ho.criterion import HeldOutMSE, FiniteDiffMonteCarloSure
from sparse_ho.ho import grad_search
from sparse_ho.optimizers import LineSearch


n_samples = 100
n_features = 100
n_active = 5
SNR = 3
rho = 0.5

X, y, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=0)
X_train_s = csc_matrix(X)

idx_train = np.arange(0, 50)
idx_val = np.arange(50, 100)

alpha_max = np.max(np.abs(X[idx_train, :].T @ y[idx_train])) / len(idx_train)
p_alpha = 0.7
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)

log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
tol = 1e-16
max_iter = 1000

dict_log_alpha0 = {}
dict_log_alpha0["lasso"] = log_alpha
tab = np.linspace(1, 1000, n_features)
dict_log_alpha0["wlasso"] = log_alpha + np.log(tab / tab.max())

models = [
    Lasso(max_iter=max_iter, estimator=None),
]

estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)
models_custom = [
    Lasso(max_iter=max_iter, estimator=estimator),
]


@pytest.mark.parametrize('model', models)
@pytest.mark.parametrize('crit', ['cv', 'sure'])
def test_grad_search(model, crit):
    """check that the paths are the same in the line search"""
    if crit == 'cv':
        n_outer = 2
        criterion = HeldOutMSE(idx_train, idx_val)
    else:
        n_outer = 2
        criterion = FiniteDiffMonteCarloSure(sigma_star)
    # TODO MM@QBE if else scheme surprising

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor1 = Monitor()
    algo = Forward()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y, log_alpha, monitor1)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor2 = Monitor()
    algo = Implicit()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(algo, criterion, model, optimizer, X, y, log_alpha, monitor2)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor3 = Monitor()
    algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=5000)
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(algo, criterion, model, optimizer, X, y, log_alpha, monitor3)

    np.testing.assert_allclose(
        np.array(monitor1.log_alphas), np.array(monitor3.log_alphas))
    np.testing.assert_allclose(
        np.array(monitor1.grads), np.array(monitor3.grads), atol=1e-8)
    np.testing.assert_allclose(
        np.array(monitor1.objs), np.array(monitor3.objs))
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor3.times))


if __name__ == '__main__':
    models = [
        Lasso(max_iter=max_iter, estimator=None)]
    crits = ['sure']
    # crits = ['cv']
    for model in models:
        for crit in crits:
            test_grad_search(model, crit)
