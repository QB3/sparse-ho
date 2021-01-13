# TODO make Backward in the test
# TODO add test for the logreg (in another file)
# TODO include tests for logreg wLasso with custom solver
# TODO add logreg and SVM here?
import pytest

import numpy as np
from scipy.optimize import check_grad
from scipy.sparse import csc_matrix
from sklearn import linear_model
import celer

from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.models import Lasso, ElasticNet, WeightedLasso, SparseLogreg

from sparse_ho import Forward, Implicit, ImplicitForward

from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.algo.implicit import get_beta_jac_t_v_implicit
from sparse_ho.criterion import (
    HeldOutMSE, FiniteDiffMonteCarloSure, HeldOutLogistic)
from sparse_ho.tests.cvxpylayer import \
    enet_cvxpy, weighted_lasso_cvxpy, logreg_cvxpy


# Generate data
n_samples, n_features = 10, 10
X, y, _, _, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=5, rho=0.1,
    SNR=3, seed=0)
y = np.sign(y)
X_s = csc_matrix(X)
idx_train = np.arange(0, n_features//2)
idx_val = np.arange(n_features//2, n_features)

# Set alpha for the Lasso
alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
p_alpha = 0.8
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)
log_alpha_max = np.log(alpha_max)
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
dict_log_alpha["logreg"] = (log_alpha - np.log(2))

# Set models to be tested
models = {}
models["lasso"] = Lasso(estimator=None)
models["enet"] = ElasticNet(estimator=None)
models["wLasso"] = WeightedLasso(estimator=None)
models["logreg"] = SparseLogreg(estimator=None)

custom_models = {}
custom_models["lasso"] = Lasso(estimator=celer.Lasso(
    warm_start=True, fit_intercept=False))
custom_models["enet"] = ElasticNet(
    estimator=linear_model.ElasticNet(warm_start=True, fit_intercept=False))
custom_models["logreg"] = SparseLogreg(
    estimator=celer.LogisticRegression(warm_start=True, fit_intercept=False))

# list of algorithms to be tested
list_algos = [
    Forward(),
    ImplicitForward(tol_jac=1e-16, n_iter_jac=5000),
    Implicit(),
    # Backward()  # XXX to fix
]


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


@pytest.mark.parametrize('key', list(models.keys()))
def test_beta_jac(key):
    """Tests that algorithms computing the Jacobian return the same Jacobian"""
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
    """Check that using sk or celer yields the same solution as sparse ho"""
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
val_cvxpy, grad_cvxpy = weighted_lasso_cvxpy(
    X, y, np.exp(dict_log_alpha["wLasso"]), idx_train, idx_val)
dict_vals_cvxpy["wLasso"] = val_cvxpy
grad_cvxpy *= np.exp(dict_log_alpha["wLasso"])
dict_grads_cvxpy["wLasso"] = grad_cvxpy
# y01 = y.copy()
# y01[y01 == -1] = 0
val_cvxpy, grad_cvxpy = logreg_cvxpy(
    X, y, np.exp(dict_log_alpha["logreg"]), idx_train, idx_val)
dict_vals_cvxpy["logreg"] = val_cvxpy
grad_cvxpy *= np.exp(dict_log_alpha["logreg"])
dict_grads_cvxpy["logreg"] = grad_cvxpy


@pytest.mark.parametrize(
    'model_name,criterion', [
        ('lasso', 'MSE'),
        ('enet', 'MSE'),
        ('wLasso', 'MSE'),
        # ('lasso', 'SURE'),
        # ('enet', 'SURE'),
        # ('wLasso', 'SURE'),
        ('logreg', 'logistic'),
    ]
)
@pytest.mark.parametrize('algo', list_algos)
def test_val_grad(model_name, criterion, algo):
    """Check that all methods return the same gradient, comparing to cvxpylayer
    """
    if criterion == 'logistic':
        pytest.xfail("cvxpylayer seems broken for logistic")

    if criterion == 'MSE':
        criterion = HeldOutMSE(idx_train, idx_val)
    elif criterion == 'SURE':
        criterion = FiniteDiffMonteCarloSure(sigma_star)
    elif criterion == 'logistic':
        criterion = HeldOutLogistic(idx_train, idx_val)

    log_alpha = dict_log_alpha[model_name]
    model = models[model_name]

    val, grad = criterion.get_val_grad(
        model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol)

    np.testing.assert_allclose(dict_vals_cvxpy[model_name], val, atol=1e-4)
    np.testing.assert_allclose(dict_grads_cvxpy[model_name], grad, atol=1e-5)


@pytest.mark.parametrize(
    'model_name,criterion', [
        ('lasso', 'MSE'),
        ('enet', 'MSE'),
        ('wLasso', 'MSE'),
        ('logreg', 'logistic'),
    ]
)
@pytest.mark.parametrize('algo', list_algos)
def test_check_grad_sparse_ho(model_name, criterion, algo):
    """Check that all methods return a good gradient using check_grad"""
    if criterion == 'MSE':
        criterion = HeldOutMSE(idx_train, idx_val)
    elif criterion == 'SURE':
        criterion = FiniteDiffMonteCarloSure(sigma_star)
    elif criterion == 'logistic':
        criterion = HeldOutLogistic(idx_train, idx_val)

    model = models[model_name]
    log_alpha = dict_log_alpha[model_name]

    def get_val(log_alpha):
        val, grad = criterion.get_val_grad(
            model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol)
        return val

    def get_grad(log_alpha):
        val, grad = criterion.get_val_grad(
            model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol)
        return grad

    print("Check grad sparse ho")
    list_log_alphas = np.log(np.geomspace(alpha_max/4, alpha_max/40, num=5))
    for log_alpha in list_log_alphas:
        grad_error = check_grad(get_val, get_grad, [log_alpha])
        assert grad_error < 1.


def test_check_grad_logreg_cvxpy():

    pytest.xfail("cvxpylayer seems broken for logistic")

    def get_val(alpha):
        val_cvxpy, grad_cvxpy = logreg_cvxpy(
            X, y, alpha[0], idx_train, idx_val)
        return val_cvxpy

    def get_grad(alpha):
        val_cvxpy, grad_cvxpy = logreg_cvxpy(
            X, y, alpha[0], idx_train, idx_val)
        return grad_cvxpy

    from scipy.optimize import check_grad

    print("Check grad cvxpy")
    list_alphas = np.geomspace(alpha_max, alpha_max / 10, num=5)
    for alpha in list_alphas:
        grad_error = check_grad(get_val, get_grad, [alpha])
        assert grad_error < 1.


if __name__ == "__main__":
    print("#" * 30)
    for algo in list_algos:
        print("#" * 20)
        test_check_grad_sparse_ho('logreg', 'logistic', algo)
    print("#" * 30)
    test_check_grad_logreg_cvxpy()
    #     for model_name in models.keys():
    #         test_val_grad(model_name, 'logistic', algo)
    # for key in list(custom_models.keys()):
    #     test_beta_jac_custom(key)
    # for key in list(models.keys()):
    #     test_beta_jac(key)