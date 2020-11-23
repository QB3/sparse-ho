import numpy as np
from sklearn import linear_model
from sparse_ho.models import ElasticNet
from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.forward import Forward
from sparse_ho.implicit import Implicit
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from scipy.sparse import csc_matrix


n_samples = 100
n_features = 100
n_active = 5
SNR = 3
rho = 0.1

X, y, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=0)
X_s = csc_matrix(X)

idx_train = np.arange(0, 50)
idx_val = np.arange(50, 100)

alpha_max = (X[idx_train, :].T @ y[idx_train]).max() / n_samples

tol = 1e-16

p_alpha = 0.7
alpha_1 = p_alpha * alpha_max
alpha_2 = 0.01
log_alpha1 = np.log(alpha_1)
log_alpha2 = np.log(alpha_2)
max_iter = 100

model = ElasticNet(max_iter=max_iter, estimator=None)
estimator = linear_model.ElasticNet(
    alpha=(alpha_1 + alpha_2), fit_intercept=False,
    l1_ratio=alpha_1 / (alpha_1 + alpha_2),
    tol=1e-16, max_iter=max_iter)
model_custom = ElasticNet(max_iter=max_iter, estimator=estimator)


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / len(idx_val)


def test_beta_jac():
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_alpha1, log_alpha2]), tol=tol, model=model, compute_jac=True, max_iter=max_iter)

    estimator = linear_model.ElasticNet(
        alpha=(alpha_1 + alpha_2), fit_intercept=False,
        l1_ratio=alpha_1 / (alpha_1 + alpha_2),
        tol=1e-16, max_iter=max_iter)
    estimator.fit(X[idx_train, :], y[idx_train])

    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_alpha1, log_alpha2]),
        get_v, tol=tol, model=model, tol_jac=1e-16, max_iter=max_iter, niter_jac=10000)
    assert np.allclose(dense1, estimator.coef_[estimator.coef_ != 0])
    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense2)


def test_beta_jac_custom():
    supp, dense, jac = get_beta_jac_fast_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_alpha1, log_alpha2]),
        get_v, tol=tol, model=model, tol_jac=1e-16, max_iter=max_iter, niter_jac=10000)
    supp_custom, dense_custom, jac_custom = get_beta_jac_fast_iterdiff(
        X[idx_train, :], y[idx_train], np.array([log_alpha1, log_alpha2]),
        get_v, tol=tol, model=model_custom, tol_jac=1e-16, max_iter=max_iter, niter_jac=10000)

    assert np.allclose(dense, dense_custom)
    assert np.allclose(supp, supp_custom)
    assert np.allclose(dense, dense_custom)


def test_val_grad():
    #######################################################################
    # Not all methods computes the full Jacobian, but all
    # compute the gradients
    # check that the gradient returned by all methods are the same
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Forward()
    val_fwd, grad_fwd = criterion.get_val_grad(
        model, X, y, np.array([log_alpha1, log_alpha2]), algo.get_beta_jac_v,
        tol=tol)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward(tol_jac=1e-16, n_iter_jac=5000)
    val_imp_fwd, grad_imp_fwd = criterion.get_val_grad(
        model, X, y, np.array([log_alpha1, log_alpha2]), algo.get_beta_jac_v,
        tol=tol)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward(tol_jac=1e-16, n_iter_jac=5000)
    val_imp_fwd_custom, grad_imp_fwd_custom = criterion.get_val_grad(
        model, X, y, np.array([log_alpha1, log_alpha2]), algo.get_beta_jac_v,
        tol=tol)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Implicit()
    val_imp, grad_imp = criterion.get_val_grad(
        model, X, y, np.array([log_alpha1, log_alpha2]),
        algo.get_beta_jac_v, tol=tol)
    assert np.allclose(val_fwd, val_imp_fwd)
    assert np.allclose(grad_fwd, grad_imp_fwd)
    assert np.allclose(val_imp_fwd, val_imp)
    assert np.allclose(val_imp_fwd, val_imp_fwd_custom)
    # for the implcit the conjugate grad does not converge
    # hence the rtol=1e-2
    assert np.allclose(grad_imp_fwd, grad_imp, atol=1e-3)
    assert np.allclose(grad_imp_fwd, grad_imp_fwd_custom)


def test_grad_search():

    n_outer = 3
    criterion = HeldOutMSE(idx_train, idx_val)
    monitor1 = Monitor()
    algo = Forward()
    grad_search(
        algo, criterion, model, X, y, np.array([log_alpha1, log_alpha2]),
        monitor1, n_outer=n_outer, tol=1e-16)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor2 = Monitor()
    algo = Implicit()
    grad_search(
        algo, criterion, model, X, y, np.array([log_alpha1, log_alpha2]),
        monitor2, n_outer=n_outer, tol=1e-16)

    criterion = HeldOutMSE(idx_train, idx_val)
    monitor3 = Monitor()
    algo = ImplicitForward(tol_jac=1e-3, n_iter_jac=1000)
    grad_search(
        algo, criterion, model, X, y, np.array([log_alpha1, log_alpha2]),
        monitor3, n_outer=n_outer, tol=1e-16)
    [np.linalg.norm(grad) for grad in monitor1.grads]
    [np.exp(alpha) for alpha in monitor1.log_alphas]

    assert np.allclose(
        np.array(monitor1.log_alphas), np.array(monitor3.log_alphas))
    assert np.allclose(
        np.array(monitor1.grads), np.array(monitor3.grads))
    assert np.allclose(
        np.array(monitor1.objs), np.array(monitor3.objs))
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor3.times))

    assert np.allclose(
        np.array(monitor1.log_alphas), np.array(monitor2.log_alphas), atol=1e-2)
    assert np.allclose(
        np.array(monitor1.grads), np.array(monitor2.grads), atol=1e-2)
    assert np.allclose(
        np.array(monitor1.objs), np.array(monitor2.objs), atol=1e-2)
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor2.times))


if __name__ == '__main__':
    test_beta_jac()
    test_val_grad()
    test_grad_search()
    test_beta_jac_custom()
