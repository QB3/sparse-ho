# XXX: go search for Alex code to save get_beta_jac and get_beta_jac2

import pytest

import numpy as np
from scipy.sparse import csc_matrix
from sklearn import linear_model

from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.models import Lasso, WeightedLasso
from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.algo.implicit import get_beta_jac_t_v_implicit

from sparse_ho import Forward
from sparse_ho import ImplicitForward
from sparse_ho import Implicit
from sparse_ho import Backward
from sparse_ho.criterion import HeldOutMSE, FiniteDiffMonteCarloSure
from sparse_ho.wrap_cvxpylayer import enet_cvx_py


n_samples = 10
n_features = 10
n_active = 5
SNR = 3
rho = 0.1

X, y, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=0)
X_s = csc_matrix(X)

idx_train = np.arange(0, n_features//2)
idx_val = np.arange(n_features//2, n_features)

alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
p_alpha = 0.8
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)

log_alphas = np.log(alpha_max * np.geomspace(1, 0.1))
tol = 1e-16

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha
tab = np.linspace(1, 1000, n_features)
dict_log_alpha["wlasso"] = log_alpha + np.log(tab / tab.max())

models = {}
models["lasso"] = Lasso(estimator=None)
models["wlasso"] = WeightedLasso(estimator=None)


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)
models_custom = {}
models_custom["lasso"] = Lasso(estimator=estimator)
models_custom["wlasso"] = WeightedLasso(estimator=estimator)


@pytest.mark.parametrize('model', list(models.keys()))
def test_beta_jac(model):
    #########################################################################
    # check that the methods computing the full Jacobian compute the same sol
    # maybe we could add a test comparing with sklearn
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X, y, dict_log_alpha[model], tol=tol, model=models[model])
    supp1sk, dense1sk, jac1sk = get_beta_jac_iterdiff(
        X, y, dict_log_alpha[model], tol=tol, model=models[model])
    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X, y, dict_log_alpha[model], tol=tol, model=models[model], tol_jac=tol)
    supp3, dense3, jac3 = get_beta_jac_iterdiff(
        X_s, y, dict_log_alpha[model], tol=tol,
        model=models[model])
    supp4, dense4, jac4 = get_beta_jac_fast_iterdiff(
        X_s, y, dict_log_alpha[model],
        tol=tol, model=models[model], tol_jac=tol)

    assert np.all(supp1 == supp1sk)
    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense1sk)
    assert np.allclose(dense1, dense2)
    assert np.allclose(jac1, jac2, atol=1e-6)

    assert np.all(supp2 == supp3)
    assert np.allclose(dense2, dense3)
    assert np.allclose(jac2, jac3, atol=1e-6)

    assert np.all(supp3 == supp4)
    assert np.allclose(dense3, dense4)
    assert np.allclose(jac3, jac4, atol=1e-6)

    get_beta_jac_t_v_implicit(
        X, y, dict_log_alpha[model], get_v, model=models[model])


@pytest.mark.parametrize('model', list(models.keys()))
def test_beta_jac2(model):
    #########################################################################
    # check that the methods computing the full Jacobian compute the same sol
    # maybe we could add a test comparing with sklearn
    supp, dense, jac = get_beta_jac_fast_iterdiff(
        X_s, y, dict_log_alpha[model],
        tol=tol, model=models[model], tol_jac=tol)
    supp_custom, dense_custom, jac_custom = get_beta_jac_fast_iterdiff(
        X_s, y, dict_log_alpha[model],
        tol=tol, model=models[model], tol_jac=tol)
    assert np.all(supp == supp_custom)
    assert np.allclose(dense, dense_custom)
    assert np.allclose(jac, jac_custom)


grad_cvxpy = enet_cvx_py(X, y, [np.exp(log_alpha), 0], idx_train, idx_val)
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
    np.testing.assert_allclose(grad, grad_cvxpy, rtol=1e-5)


if __name__ == "__main__":
    for algo in list_algos:
        for model in models:
            for criterion in ['MSE', 'SURE']:
                test_val_grad_mse(model, criterion, algo)
