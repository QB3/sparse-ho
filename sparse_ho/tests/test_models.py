# TODO make Backward in the test
# TODO include tests for wLasso with custom solver
import pytest

import numpy as np
from scipy.optimize import check_grad

from sparse_ho import Forward, ImplicitForward, Implicit

from sparse_ho.algo.forward import compute_beta
from sparse_ho.algo.implicit_forward import (get_bet_jac_implicit_forward,
                                             get_only_jac)
from sparse_ho.algo.implicit import compute_beta_grad_implicit
from sparse_ho.criterion import (
    HeldOutMSE, FiniteDiffMonteCarloSure, HeldOutLogistic)

from sparse_ho.tests.common import (
    X, X_s, y, sigma_star, idx_train, idx_val,
    dict_log_alpha, models, custom_models, dict_cvxpy_func,
    dict_vals_cvxpy, dict_grads_cvxpy, dict_list_log_alphas, get_grad_outer,
    list_model_crit, list_model_names)

# list of algorithms to be tested
list_algos = [
    Forward(),
    ImplicitForward(tol_jac=1e-8, n_iter_jac=5000),
    Implicit()
    # Backward()  # XXX to fix
]

tol = 1e-14
X_r = X_s.tocsr()
X_c = X_s


@pytest.mark.parametrize('key', list(models.keys()))
def test_beta_jac(key):
    """Tests that algorithms computing the Jacobian return the same Jacobian"""
    if key == 'svm':
        return True
    if key == "svm" or key == "svr" or key == "ssvr":
        X_s = X_r
    else:
        X_s = X_c
    supp1, dense1, jac1 = compute_beta(
        X, y, dict_log_alpha[key], tol=tol, model=models[key])
    supp2, dense2, jac2 = get_bet_jac_implicit_forward(
        X, y, dict_log_alpha[key], tol=tol, model=models[key], tol_jac=tol)
    supp3, dense3, jac3 = compute_beta(
        X_s, y, dict_log_alpha[key], tol=tol,
        model=models[key])
    supp4, dense4, jac4 = get_bet_jac_implicit_forward(
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

    compute_beta_grad_implicit(
        X, y, dict_log_alpha[key], get_grad_outer, model=models[key])


@pytest.mark.parametrize('model_name', list(custom_models.keys()))
def test_beta_jac_custom(model_name):
    """Check that using sk or celer yields the same solution as sparse ho"""
    if model_name in ("svm", "svr", "ssvr"):
        X_s = X_r
    else:
        X_s = X_c

    for log_alpha in dict_list_log_alphas[model_name]:
        supp, dense, jac = get_bet_jac_implicit_forward(
            X_s, y, log_alpha,
            tol=tol, model=models[model_name], tol_jac=tol)
        supp_custom, dense_custom, jac_custom = get_bet_jac_implicit_forward(
            X_s, y, log_alpha,
            tol=tol, model=custom_models[model_name], tol_jac=tol)
        assert np.all(supp == supp_custom)
        assert np.allclose(dense, dense_custom)
        assert np.allclose(jac, jac_custom)


@pytest.mark.parametrize('model_name', list(custom_models.keys()))
def test_warm_start(model_name):
    """Check that warm start leads to only 2 iterations
    in Jacobian computation"""
    if model_name in ("svm", "svr", "ssvr"):
        X_s = X_r
    else:
        X_s = X_c
    model = models[model_name]

    for log_alpha in dict_list_log_alphas[model_name]:
        mask, dense, jac = None, None, None
        for i in range(2):
            mask, dense, _ = compute_beta(
                X_s, y, log_alpha, tol=tol,
                mask0=mask, dense0=dense, jac0=jac,
                max_iter=5000, compute_jac=False, model=model)
            dbeta0_new = model._init_dbeta0(mask, mask, jac)
            reduce_alpha = model._reduce_alpha(np.exp(log_alpha), mask)

            _, dual_var = model._init_beta_dual_var(X_s, y, mask, dense)
            jac = get_only_jac(
                model.reduce_X(X_s, mask), model.reduce_y(y, mask), dual_var,
                reduce_alpha, model.sign(dense, log_alpha), dbeta=dbeta0_new,
                niter_jac=5000, tol_jac=1e-13, model=model, mask=mask,
                dense=dense)
            if i == 0:
                np.testing.assert_array_less(2, get_only_jac.n_iter)
            else:
                assert get_only_jac.n_iter == 2


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
        model, X, y, log_alpha, algo.compute_beta_grad, tol=tol)
    np.testing.assert_allclose(
        dict_vals_cvxpy[model_name, criterion_name], val, rtol=1e-3, atol=1e-3)
    np.testing.assert_allclose(
        dict_grads_cvxpy[model_name, criterion_name], grad,
        rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize('model_name,criterion', list_model_crit)
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
            model, X, y, np.squeeze(log_alpha), algo.compute_beta_grad,
            tol=tol)
        return val

    def get_grad(log_alpha):
        _, grad = criterion.get_val_grad(
            model, X, y, np.squeeze(log_alpha), algo.compute_beta_grad,
            tol=tol)
        return grad

    for log_alpha in dict_list_log_alphas[model_name]:
        grad_error = check_grad(get_val, get_grad, log_alpha)
        assert grad_error < 1e-1


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
        test_val_grad("lasso", "MSE", algo)
        test_check_grad_sparse_ho('lasso', 'MSE', algo)
        test_beta_jac('lasso')
