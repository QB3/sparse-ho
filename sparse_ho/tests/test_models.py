# TODO include test for the wLasso here
import pytest
from scipy.sparse import csc_matrix

import numpy as np
from sklearn import linear_model
import celer

from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.models import Lasso, ElasticNet, WeightedLasso

from sparse_ho import Forward
from sparse_ho import ImplicitForward
# from sparse_ho import Backward

from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.algo.implicit import get_beta_jac_t_v_implicit
from sparse_ho.criterion import HeldOutMSE, FiniteDiffMonteCarloSure
from sparse_ho.wrap_cvxpylayer import enet_cvxpy, wLasso_cvxpy


# Generate data
n_samples = 10
n_features = 10
X, y, _, _, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=5, rho=0.1,
    SNR=3, seed=0)
X_s = csc_matrix(X)
idx_train = np.arange(0, n_features//2)
idx_val = np.arange(n_features//2, n_features)

# Set alpha for the Lasso
alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
p_alpha = 0.8
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)
tol = 1e-16

# Set alpha1 alpha2 for the enet
alpha_1 = p_alpha * alpha_max
alpha_2 = 0.1
log_alpha1 = np.log(alpha_1)
log_alpha2 = np.log(alpha_2)

dict_log_alpha = {}
dict_log_alpha["lasso"] = log_alpha
dict_log_alpha["enet"] = np.array([log_alpha1, log_alpha2])
tab = np.linspace(1, 1000, n_features)
dict_log_alpha["wLasso"] = log_alpha + np.log(tab / tab.max())

# Set models to be tested
models = {}
models["lasso"] = Lasso(estimator=None)
models["enet"] = ElasticNet(estimator=None)
models["wLasso"] = WeightedLasso(estimator=None)
custom_models = {}
custom_models["lasso"] = Lasso(estimator=celer.Lasso(
    warm_start=True, fit_intercept=False))
custom_models["enet"] = ElasticNet(
    estimator=linear_model.ElasticNet(warm_start=True, fit_intercept=False))

# list of algorithms to be tested
list_algos = [
    Forward(),
    ImplicitForward(tol_jac=1e-16, n_iter_jac=5000)]
# Backward()]
# TODO make Backward pass
# TODO add test for the logreg


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


@pytest.mark.parametrize('key', list(models.keys()))
def test_beta_jac(key):
    """
    Tests that the algorithms computing the Jacobian return the same Jacobian
    """
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X, y, dict_log_alpha[key], tol=tol, model=models[key])
    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X, y, dict_log_alpha[key], tol=tol, model=models[key], tol_jac=tol)
    supp3, dense3, jac3 = get_beta_jac_iterdiff(
        X_s, y, dict_log_alpha[key], tol=tol,
        model=models[key])
    supp4, dense4, jac4 = get_beta_jac_fast_iterdiff(
        X_s, y, dict_log_alpha[key],
        tol=tol, model=models[key], tol_jac=tol)

    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense2)
    assert np.allclose(jac1, jac2, atol=1e-6)

    assert np.all(supp2 == supp3)
    assert np.allclose(dense2, dense3)
    assert np.allclose(jac2, jac3, atol=1e-6)

    assert np.all(supp3 == supp4)
    assert np.allclose(dense3, dense4)
    assert np.allclose(jac3, jac4, atol=1e-6)

    get_beta_jac_t_v_implicit(
        X, y, dict_log_alpha[key], get_v, model=models[key])


@pytest.mark.parametrize('key', list(custom_models.keys()))
def test_beta_jac_custom(key):
    """Check that using sk or celer yields the same solution as sparse ho
    """
    supp, dense, jac = get_beta_jac_fast_iterdiff(
        X_s, y, dict_log_alpha[key],
        tol=tol, model=models[key], tol_jac=tol)
    supp_custom, dense_custom, jac_custom = get_beta_jac_fast_iterdiff(
        X_s, y, dict_log_alpha[key],
        tol=tol, model=custom_models[key], tol_jac=tol)
    assert np.all(supp == supp_custom)
    assert np.allclose(dense, dense_custom)
    assert np.allclose(jac, jac_custom)


# Compute "ground truth" with cvxpylayer
dict_vals_cvxpy = {}
dict_grads_cvxpy = {}
val_cvxpy, grad_cvxpy = enet_cvxpy(
    X, y, [np.exp(log_alpha), 0], idx_train, idx_val)
dict_vals_cvxpy["lasso"] = val_cvxpy
grad_cvxpy *= np.exp(log_alpha)
grad_cvxpy = grad_cvxpy[0]
dict_grads_cvxpy["lasso"] = grad_cvxpy
val_cvxpy, grad_cvxpy = enet_cvxpy(
    X, y, np.exp(dict_log_alpha["enet"]), idx_train, idx_val)
dict_vals_cvxpy["enet"] = val_cvxpy
grad_cvxpy *= np.exp(dict_log_alpha["enet"])
dict_grads_cvxpy["enet"] = grad_cvxpy
val_cvxpy, grad_cvxpy = wLasso_cvxpy(
    X, y, np.exp(dict_log_alpha["wLasso"]), idx_train, idx_val)
dict_vals_cvxpy["wLasso"] = val_cvxpy
grad_cvxpy *= np.exp(dict_log_alpha["wLasso"])
dict_grads_cvxpy["wLasso"] = grad_cvxpy


@pytest.mark.parametrize('model_name', list(models.keys()))
@pytest.mark.parametrize('criterion', ['MSE'])
@pytest.mark.parametrize('algo', list_algos)
def test_val_grad_mse(model_name, criterion, algo):
    """Check that all methods return the same gradient, comparing to cvxpylayer
    """
    if criterion == 'MSE':
        criterion = HeldOutMSE(idx_train, idx_val)
    elif criterion == 'SURE':
        criterion = FiniteDiffMonteCarloSure(sigma_star)

    log_alpha = dict_log_alpha[model_name]
    model = models[model_name]

    val, grad = criterion.get_val_grad(
        model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol)

    np.testing.assert_allclose(dict_vals_cvxpy[model_name], val, atol=1e-6)
    np.testing.assert_allclose(dict_grads_cvxpy[model_name], grad, atol=1e-5)


if __name__ == "__main__":
    for algo in list_algos:
        for model in models:
            for criterion in ['MSE']:
                test_val_grad_mse(model, criterion, algo)
    for key in list(custom_models.keys()):
        test_beta_jac_custom(key)
    for key in list(models.keys()):
        test_beta_jac(key)
