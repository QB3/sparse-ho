# TODO make Backward in the test
# TODO include tests for wLasso with custom solver
# TODO add SVM and SVR here -->> We will do it with QK
import pytest

import numpy as np
from scipy.optimize import check_grad

from sparse_ho import Forward, ImplicitForward, Implicit

from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.algo.implicit import get_beta_jac_t_v_implicit
from sparse_ho.criterion import (
    HeldOutMSE, FiniteDiffMonteCarloSure, HeldOutLogistic)

from sparse_ho.tests.common import (
    X, X_s, y, sigma_star, idx_train, idx_val,
    dict_log_alpha, models, custom_models, dict_cvxpy_func,
    dict_vals_cvxpy, dict_grads_cvxpy, dict_list_log_alphas, get_v,
    list_model_crit)

# list of algorithms to be tested
list_algos = [
    Forward(),
    ImplicitForward(tol_jac=1e-16, n_iter_jac=5000),
    Implicit()
    # Backward()  # XXX to fix
]

tol = 1e-15


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


@pytest.mark.parametrize('model_name', list(custom_models.keys()))
def test_beta_jac_custom(model_name):
    """Check that using sk or celer yields the same solution as sparse ho"""
    for log_alpha in dict_list_log_alphas[model_name]:
        supp, dense, jac = get_beta_jac_fast_iterdiff(
            X_s, y, log_alpha,
            tol=tol, model=models[model_name], tol_jac=tol)
        supp_custom, dense_custom, jac_custom = get_beta_jac_fast_iterdiff(
            X_s, y, log_alpha,
            tol=tol, model=custom_models[model_name], tol_jac=tol)
        assert np.all(supp == supp_custom)
        assert np.allclose(dense, dense_custom)
        assert np.allclose(jac, jac_custom)


@pytest.mark.parametrize('model_name,criterion_name', list_model_crit)
@pytest.mark.parametrize('algo', list_algos)
def test_val_grad(model_name, criterion_name, algo):
    """Check that all methods return the same gradient, comparing to cvxpylayer
    """
    if criterion_name == 'logistic':
        pytest.xfail("cvxpylayer seems broken for logistic")

    if criterion_name == 'MSE':
        criterion = HeldOutMSE(idx_train, idx_val)
    elif criterion_name == 'logistic':
        criterion = HeldOutLogistic(idx_train, idx_val)
    elif criterion_name == 'SURE':
        criterion = FiniteDiffMonteCarloSure(sigma_star)

    log_alpha = dict_log_alpha[model_name]
    model = models[model_name]

    val, grad = criterion.get_val_grad(
        model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol)

    np.testing.assert_allclose(
        dict_vals_cvxpy[model_name, criterion_name], val, rtol=1e-5)
    np.testing.assert_allclose(
        dict_grads_cvxpy[model_name, criterion_name], grad,
        rtol=1e-5, atol=1e-5)


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
        val, _ = criterion.get_val_grad(
            model, X, y, np.squeeze(log_alpha), algo.get_beta_jac_v, tol=tol)
        return val

    def get_grad(log_alpha):
        _, grad = criterion.get_val_grad(
            model, X, y, np.squeeze(log_alpha), algo.get_beta_jac_v, tol=tol)
        return grad

    for log_alpha in dict_list_log_alphas[model_name]:
        grad_error = check_grad(get_val, get_grad, log_alpha)
        assert grad_error < 1e-1


list_model_names = ["lasso", "enet", "wLasso", "logreg"]


@pytest.mark.parametrize('model_name', list_model_names)
def test_check_grad_logreg_cvxpy(model_name):

    pytest.xfail("cvxpylayer seems broken for logistic")
    cvxpy_func = dict_cvxpy_func[model_name]

    def get_val(log_alpha):
        val_cvxpy, _ = cvxpy_func(
            X, y, np.exp(log_alpha), idx_train, idx_val)
        return val_cvxpy

    def get_grad(log_alpha):
        _, grad_cvxpy = cvxpy_func(
            X, y, np.exp(log_alpha), idx_train, idx_val)
        grad_cvxpy *= np.exp(log_alpha)
        return grad_cvxpy

    for log_alpha in dict_list_log_alphas[model_name]:
        grad_error = check_grad(get_val, get_grad, log_alpha)
        assert grad_error < 1


if __name__ == "__main__":
    test_beta_jac_custom("logreg")
    print("#" * 30)
    for algo in list_algos:
        print("#" * 20)
        test_val_grad("lasso", "SURE", algo)
        test_check_grad_sparse_ho('lasso', 'MSE', algo)
        test_check_grad_sparse_ho('enet', 'MSE', algo)
