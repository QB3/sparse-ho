import numpy as np
from sklearn import linear_model
from sparse_ho.models import ElasticNet
from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.datasets.synthetic import get_synt_data
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.criterion import CV
from sparse_ho.forward import Forward
from sparse_ho.implicit import Implicit
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from scipy.sparse import csc_matrix


n_samples = 10
n_features = 20
n_active = 5
tol = 1e-16
max_iter = 50000
SNR = 3
rho = 0.1
X_train, y_train, beta_star, noise, sigma_star = get_synt_data(
    dictionary_type="Gaussian", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=0)
X_train = csc_matrix(X_train)
X_test, y_test, beta_star, noise, sigma = get_synt_data(
    dictionary_type="Gaussian", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=1)
X_test = csc_matrix(X_test)
X_val, y_val, beta_star, noise, sigma = get_synt_data(
    dictionary_type="Gaussian", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, rho=rho,
    SNR=SNR, seed=2)
X_val = csc_matrix(X_val)
alpha_max = (X_train.T @ y_train).max() / n_samples
p_alpha = 0.7
alpha_1 = p_alpha * alpha_max
alpha_2 = 0.01
log_alpha1 = np.log(alpha_1)
log_alpha2 = np.log(alpha_2)

model = ElasticNet(X_train, y_train, max_iter=max_iter, estimator=None)
estimator = linear_model.ElasticNet(
    alpha=(alpha_1 + alpha_2), fit_intercept=False,
    l1_ratio=alpha_1 / (alpha_1 + alpha_2),
    tol=1e-16, max_iter=max_iter)
model_custom = ElasticNet(X_train, y_train, max_iter=max_iter, estimator=estimator)


def get_v(mask, dense):
    return 2 * (X_val[:, mask].T @ (
        X_val[:, mask] @ dense - y_val)) / X_val.shape[0]


def test_beta_jac():
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X_train, y_train, np.array([log_alpha1, log_alpha2]), tol=tol,
        model=model, compute_jac=True, max_iter=max_iter)

    estimator = linear_model.ElasticNet(
        alpha=(alpha_1 + alpha_2), fit_intercept=False,
        l1_ratio=alpha_1 / (alpha_1 + alpha_2),
        tol=1e-16, max_iter=max_iter)
    estimator.fit(X_train, y_train)

    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X_train, y_train, np.array([log_alpha1, log_alpha2]), tol=tol,
        model=model, tol_jac=1e-16, max_iter=max_iter, niter_jac=10000)
    assert np.allclose(dense1, estimator.coef_[estimator.coef_ != 0])
    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense2)


def test_beta_jac_custom():
    supp, dense, jac = get_beta_jac_fast_iterdiff(
        X_train, y_train, np.array([log_alpha1, log_alpha2]),
        tol=tol, model=model, tol_jac=1e-16, max_iter=max_iter, niter_jac=10000)
    supp_custom, dense_custom, jac_custom = get_beta_jac_fast_iterdiff(
        X_train, y_train, np.array([log_alpha1, log_alpha2]),
        tol=tol, model=model_custom, tol_jac=1e-16, max_iter=max_iter, niter_jac=10000)

    assert np.allclose(dense, dense_custom)
    assert np.allclose(supp, supp_custom)
    assert np.allclose(dense, dense_custom)


def test_val_grad():
    #######################################################################
    # Not all methods computes the full Jacobian, but all
    # compute the gradients
    # check that the gradient returned by all methods are the same
    criterion = CV(X_val, y_val, model)
    algo = Forward(criterion)
    val_fwd, grad_fwd = algo.get_val_grad(
        np.array([log_alpha1, log_alpha2]), tol=tol)

    criterion = CV(X_val, y_val, model)
    algo = ImplicitForward(criterion, tol_jac=1e-16, n_iter_jac=5000)
    val_imp_fwd, grad_imp_fwd = algo.get_val_grad(
        np.array([log_alpha1, log_alpha2]), tol=tol)

    criterion = CV(X_val, y_val, model)
    algo = ImplicitForward(
        criterion, tol_jac=1e-16, n_iter_jac=5000)
    val_imp_fwd_custom, grad_imp_fwd_custom = algo.get_val_grad(
        np.array([log_alpha1, log_alpha2]), tol=tol)

    criterion = CV(X_val, y_val, model)
    algo = Implicit(criterion)
    val_imp, grad_imp = algo.get_val_grad(
        np.array([log_alpha1, log_alpha2]), tol=tol)
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
    criterion = CV(X_val, y_val, model, X_test=None, y_test=None)
    monitor1 = Monitor()
    algo = Forward(criterion)
    grad_search(
        algo, np.array([log_alpha1, log_alpha2]), monitor1, n_outer=n_outer,
        tol=1e-16)

    criterion = CV(X_val, y_val, model, X_test=None, y_test=None)
    monitor2 = Monitor()
    algo = Implicit(criterion)
    grad_search(
        algo, np.array([log_alpha1, log_alpha2]), monitor2, n_outer=n_outer,
        tol=1e-16)

    criterion = CV(X_val, y_val, model, X_test=None, y_test=None)
    monitor3 = Monitor()
    algo = ImplicitForward(criterion, tol_jac=1e-3, n_iter_jac=1000)
    grad_search(
        algo, np.array([log_alpha1, log_alpha2]), monitor3, n_outer=n_outer,
        tol=1e-16)
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
