import numpy as np
from scipy.sparse import csc_matrix
from sklearn import svm

from sparse_ho.models import SVR
from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.datasets import get_synt_data
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import Forward
from sparse_ho import Implicit
from sparse_ho import ImplicitForward
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor

from sparse_ho.optimizers import LineSearch

n_samples = 25
n_features = 5
n_active = 5
SNR = 100

X, y, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Gaussian", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=0.0,
    SNR=SNR, seed=0)
X_s = csc_matrix(X)

idx_train = np.arange(0, 12)
idx_val = np.arange(12, 25)

tol = 1e-16

C = 0.1
log_C = np.log(C)
log_epsilon = np.log(0.1)
max_iter = 100000

model = SVR(max_iter=max_iter, estimator=None)
estimator = svm.LinearSVR(
    epsilon=np.exp(log_epsilon), tol=1e-16, C=np.exp(log_C),
    fit_intercept=False, max_iter=1000)
model_custom = SVR(max_iter=max_iter, estimator=estimator)


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


def test_beta_jac():
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_C, log_epsilon]),
        tol=tol, model=model, compute_jac=True, max_iter=max_iter)

    estimator = svm.LinearSVR(
        epsilon=np.exp(log_epsilon), tol=1e-16, C=np.exp(log_C),
        fit_intercept=False, max_iter=10000)
    estimator.fit(X[idx_train, :], y[idx_train])
    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_C, log_epsilon]),
        tol=tol, model=model, tol_jac=1e-8, max_iter=max_iter,
        niter_jac=10000)

    assert np.allclose(dense1, estimator.coef_[estimator.coef_ != 0])
    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense2)


# def test_beta_jac_custom():
#     supp, dense, jac = get_beta_jac_fast_iterdiff(
#         X[idx_train, :], y[idx_train], np.array([log_C, log_epsilon]),
#         tol=tol, model=model, tol_jac=1e-8, max_iter=max_iter,
#         niter_jac=10000)
#     supp_custom, dense_custom, jac_custom = get_beta_jac_fast_iterdiff(
#         X[idx_train, :], y[idx_train], np.array([log_C, log_epsilon]),
#         tol=tol, model=model_custom, tol_jac=1e-8, max_iter=max_iter,
#         niter_jac=10000)

#     assert np.allclose(dense, dense_custom)
#     assert np.allclose(supp, supp_custom)
#     assert np.allclose(dense, dense_custom)


def test_val_grad():
    #######################################################################
    # Not all methods computes the full Jacobian, but all
    # compute the gradients
    # check that the gradient returned by all methods are the same
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Forward()
    val_fwd, grad_fwd = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]), algo.get_beta_jac_v,
        tol=tol, max_iter=max_iter)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward(tol_jac=1e-16, n_iter_jac=10000)
    val_imp_fwd, grad_imp_fwd = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]), algo.get_beta_jac_v,
        tol=tol, max_iter=max_iter)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward(tol_jac=1e-16, n_iter_jac=10000)
    val_imp_fwd_custom, grad_imp_fwd_custom = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]), algo.get_beta_jac_v,
        tol=tol, max_iter=max_iter)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Implicit()
    val_imp, grad_imp = criterion.get_val_grad(
        model, X, y, np.array([log_C, log_epsilon]),
        algo.get_beta_jac_v, tol=tol)
    assert np.allclose(val_fwd, val_imp_fwd)
    assert np.allclose(grad_fwd, grad_imp_fwd)
    assert np.allclose(val_imp_fwd, val_imp)
    assert np.allclose(grad_imp_fwd, grad_imp, atol=1e-5)


def test_grad_search():

    n_outer = 3
    criterion = HeldOutMSE(idx_train, idx_val)
    monitor1 = Monitor()
    algo = Forward()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y,
        np.array([log_C, log_epsilon]), monitor1)
    criterion = HeldOutMSE(idx_train, idx_val)
    monitor2 = Monitor()
    algo = Implicit()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y, np.array(
            [log_C, log_epsilon]), monitor2)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor3 = Monitor()
    algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=1000)
    optimizer = LineSearch(n_outer=n_outer, tol=1e-16)
    grad_search(
        algo, criterion, model, optimizer, X, y,
        np.array([log_C, log_epsilon]), monitor3)
    [np.linalg.norm(grad) for grad in monitor1.grads]
    [np.exp(alpha) for alpha in monitor1.log_alphas]

    assert np.allclose(
        np.array(monitor1.log_alphas), np.array(monitor3.log_alphas))
    assert np.allclose(
        np.array(monitor1.grads), np.array(monitor3.grads), atol=1e-2)
    assert np.allclose(
        np.array(monitor1.objs), np.array(monitor3.objs), atol=1e-2)
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor3.times))

    np.testing.assert_allclose(
        np.array(monitor1.log_alphas), np.array(monitor2.log_alphas),
        atol=1e-2)
    np.testing.assert_allclose(
        np.array(monitor1.grads), np.array(monitor2.grads), atol=1e-2)
    np.testing.assert_allclose(
        np.array(monitor1.objs), np.array(monitor2.objs), atol=1e-2)
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor2.times))


if __name__ == '__main__':
    test_beta_jac()
    test_val_grad()
    test_grad_search()
    # test_beta_jac_custom()
