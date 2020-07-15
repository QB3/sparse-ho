import numpy as np
from sklearn.svm import LinearSVR
from sparse_ho.models import SVR
from sparse_ho.forward import get_beta_jac_iterdiff
from sklearn.datasets import make_regression
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.criterion import CV
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor


n_samples = 100
n_features = 100
n_active = 100
SNR = 5.0
tol = 1e-16
C = 1.0
log_C = np.log(C)
epsilon = 0.05
log_epsilon = np.log(epsilon)

max_iter = 5000
X_train, y_train, beta_star = make_regression(
    shuffle=False, random_state=15, n_samples=n_samples, n_features=n_features,
    n_informative=n_features, n_targets=1, coef=True)

X_val, y_val, beta_star = make_regression(
    shuffle=False, random_state=125, n_samples=n_samples, n_features=n_features,
    n_informative=n_features, n_targets=1, coef=True)

model = SVR(X_train, y_train, log_C, log_epsilon, max_iter=max_iter, tol=tol)


def get_v(mask, dense):
    return 2 * (X_val[:, mask].T @ (
        X_val[:, mask] @ dense - y_val)) / X_val.shape[0]


def test_beta_jac():
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X_train, y_train, np.array([log_C, log_epsilon]), tol=tol,
        model=model, compute_jac=True, max_iter=max_iter)

    dual = np.zeros(2 * n_samples)
    dual[supp1] = dense1
    primal = X_train.T @ (dual[0:n_samples] - dual[n_samples:(2 * n_samples)])
    clf = LinearSVR(
        epsilon=epsilon, fit_intercept=False, C=C, tol=1e-12, max_iter=max_iter)
    clf.fit(X_train, y_train)

    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X_train, y_train, np.array([log_C, log_epsilon]), None, None,
        get_v, tol=tol, model=model, tol_jac=1e-1, max_iter=max_iter, niter_jac=10000)

    assert np.allclose(primal, clf.coef_)
    assert np.allclose(dense1, dense2)
    assert np.all(supp1 == supp2)
    assert np.allclose(jac1, jac2, atol=1e-3)


def test_val_grad():
    #######################################################################
    # Not all methods computes the full Jacobian, but all
    # compute the gradients
    # check that the gradient returned by all methods are the same

    criterion = CV(X_val, y_val, model)
    algo = Forward(criterion)
    val_fwd, grad_fwd = algo.get_val_grad(
        np.array([log_C, log_epsilon]), tol=tol)

    criterion = CV(X_val, y_val, model)
    algo = ImplicitForward(criterion, tol_jac=1e-2, n_iter_jac=1000)
    val_imp_fwd, grad_imp_fwd = algo.get_val_grad(
        np.array([log_C, log_epsilon]), tol=tol)
    assert np.allclose(val_fwd, val_imp_fwd)
    assert np.allclose(grad_fwd, grad_imp_fwd, atol=1e-3)


def test_grad_search():

    n_outer = 3
    criterion = CV(X_val, y_val, model, X_test=None, y_test=None)
    monitor1 = Monitor()
    algo = Forward(criterion)
    grad_search(algo, np.array([np.log(1e-1), log_epsilon]), monitor1, n_outer=n_outer,
                tol=1e-13)

    criterion = CV(X_val, y_val, model, X_test=None, y_test=None)
    monitor3 = Monitor()
    algo = ImplicitForward(criterion, tol_jac=1e-6, n_iter_jac=1000)
    grad_search(algo, np.array([np.log(1e-1), log_epsilon]), monitor3, n_outer=n_outer,
                tol=1e-13)

    # import ipdb; ipdb.set_trace()
    # assert np.allclose(
    #     np.array(monitor1.log_alphas), np.array(monitor3.log_alphas))
    # assert np.allclose(
    #     np.array(monitor1.grads), np.array(monitor3.grads))
    # assert np.allclose(
    #     np.array(monitor1.objs), np.array(monitor3.objs))
    # assert not np.allclose(
    #     np.array(monitor1.times), np.array(monitor3.times))


if __name__ == '__main__':
    test_beta_jac()
    test_val_grad()
    test_grad_search()
