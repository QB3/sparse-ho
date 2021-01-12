import pytest

import numpy as np
from sklearn import linear_model

from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.models import Lasso

from sparse_ho import Forward
from sparse_ho import ImplicitForward
from sparse_ho import Backward
from sparse_ho.criterion import HeldOutMSE, FiniteDiffMonteCarloSure
from sparse_ho.wrap_cvxpylayer import enet_cvx_py


n_samples = 10
n_features = 10
X, y, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=5, rho=0.1,
    SNR=3, seed=0)

idx_train = np.arange(0, n_features//2)
idx_val = np.arange(n_features//2, n_features)

alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
p_alpha = 0.8
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)

tol = 1e-16

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)


models = {}
models["lasso"] = Lasso(estimator=None)
# models["lasso_custom"] = Lasso(estimator=estimator)


val_cvxpy, grad_cvxpy = enet_cvx_py(X, y, [np.exp(log_alpha), 0], idx_train, idx_val)
grad_cvxpy *= np.exp(log_alpha)
grad_cvxpy = grad_cvxpy[0]

list_algos = [
    Forward(),
    ImplicitForward(tol_jac=1e-16, n_iter_jac=5000),
    Backward()]

@pytest.mark.pararmetrize('algo', list_algos)
@pytest.mark.parametrize('model', list(models.keys()))
@pytest.mark.parametrize('criterion', ['MSE'])
def test_val_grad_mse(model, criterion, algo):

    if criterion == 'MSE':
        criterion = HeldOutMSE(idx_train, idx_val)
    elif criterion == 'SURE':
        criterion = FiniteDiffMonteCarloSure(sigma_star)

    #######################################################################
    # Not all methods computes the full Jacobian, but all
    # compute the gradients
    # check that the gradient returned by all methods are the same
    log_alpha = dict_log_alpha[model]
    model = models[model]

    val, grad = criterion.get_val_grad(
        model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol)

    import ipdb; ipdb.set_trace()
    np.testing.assert_allclose(val, val_cvxpy, rtol=1e-5)
    np.testing.assert_allclose(grad, grad_cvxpy, rtol=1e-5)


if __name__ == "__main__":
    for algo in list_algos:
        for model in models:
            for criterion in ['MSE']:
                test_val_grad_mse(model, criterion, algo)
