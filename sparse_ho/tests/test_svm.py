import numpy as np
import pytest

from sklearn import datasets
from sklearn.svm import LinearSVC

from sparse_ho.criterion import HeldOutLogistic
from sparse_ho.models import SVM
from sparse_ho.algo import Forward
from sparse_ho.algo import Implicit
from sparse_ho.algo import ImplicitForward
from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.ho import grad_search
from sparse_ho.criterion import HeldOutSmoothedHinge
from sparse_ho.utils import Monitor

from sparse_ho.optimizers import LineSearch

n_samples = 100
n_features = 300

X, y = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features, n_informative=50,
    random_state=11, flip_y=0.1, n_redundant=0)

y[y == 0.0] = -1.0
idx_train = np.arange(0, 50)
idx_val = np.arange(50, 100)


C = 0.001
log_C = np.log(C)
tol = 1e-16

models = [SVM(max_iter=10000)]


def get_v(mask, dense):
    return 2 * (X[np.ix_(idx_val, mask)].T @ (
        X[np.ix_(idx_val, mask)] @ dense - y[idx_val])) / idx_val.shape[0]


@pytest.mark.parametrize('model', models)
def test_beta_jac(model):
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X[idx_train, :], y[idx_train], log_C, tol=tol,
        model=model, compute_jac=True, max_iter=10000)
    clf = LinearSVC(
        loss="hinge", fit_intercept=False, C=C, tol=tol, max_iter=100000)
    clf.fit(X[idx_train, :], y[idx_train])
    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X[idx_train, :], y[idx_train], log_C,
        tol=tol, model=model, tol_jac=1e-16, max_iter=10000)
    assert np.allclose(dense1, clf.coef_[clf.coef_ != 0])

    assert np.all(supp1 == supp2)
    np.testing.assert_allclose(dense1, dense2)
    np.testing.assert_allclose(jac1, jac2, atol=1e-4)


@pytest.mark.parametrize('model', models)
def test_val_grad(model):
    #######################################################################
    # Not all methods computes the full Jacobian, but all
    # compute the gradients
    # check that the gradient returned by all methods are the same

    criterion = HeldOutLogistic(idx_train, idx_val)
    algo = Forward()
    val_fwd, grad_fwd = criterion.get_val_grad(
        model, X, y, log_C, algo.get_beta_jac_v, tol=tol)

    criterion = HeldOutLogistic(idx_train, idx_val)
    algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=1000)
    val_imp_fwd, grad_imp_fwd = criterion.get_val_grad(
        model, X, y, log_C, algo.get_beta_jac_v, tol=tol)

    criterion = HeldOutLogistic(idx_train, idx_val)
    algo = Implicit()
    val_imp, grad_imp = criterion.get_val_grad(
        model, X, y, log_C, algo.get_beta_jac_v, tol=tol)

    np.testing.assert_allclose(val_fwd, val_imp_fwd)
    np.testing.assert_allclose(grad_fwd, grad_imp_fwd)
    np.testing.assert_allclose(val_imp_fwd, val_imp)
    np.testing.assert_allclose(grad_imp_fwd, grad_imp, atol=1e-5)


@pytest.mark.parametrize('model', models)
def test_grad_search(model):
    n_outer = 3
    criterion = HeldOutSmoothedHinge(idx_train, idx_val)
    monitor1 = Monitor()
    algo = Forward()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-13)
    grad_search(
        algo, criterion, model, optimizer, X, y, np.log(1e-3), monitor1)

    criterion = HeldOutSmoothedHinge(idx_train, idx_val)
    monitor2 = Monitor()
    algo = Implicit()
    optimizer = LineSearch(n_outer=n_outer, tol=1e-13)
    grad_search(
        algo, criterion, model, optimizer, X, y, np.log(1e-3), monitor2)

    criterion = HeldOutSmoothedHinge(idx_train, idx_val)
    monitor3 = Monitor()
    algo = ImplicitForward(tol_jac=1e-6, n_iter_jac=100)
    grad_search(
        algo, criterion, model, optimizer, X, y, np.log(1e-3), monitor3)

    np.testing.assert_allclose(
        np.array(monitor1.log_alphas), np.array(monitor3.log_alphas))
    np.testing.assert_allclose(
        np.array(monitor1.grads), np.array(monitor3.grads), atol=1e-8)
    np.testing.assert_allclose(
        np.array(monitor1.objs), np.array(monitor3.objs))
    # np.testing.assert_allclose(
    #     np.array(monitor1.objs_test), np.array(monitor3.objs_test))
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
    for model in models:
        test_beta_jac(model)
        test_val_grad(model)
        test_grad_search(model)
