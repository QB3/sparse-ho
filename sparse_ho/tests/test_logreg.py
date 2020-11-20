import numpy as np
import pytest
from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from scipy.sparse import csc_matrix

from sparse_ho.models import SparseLogreg
from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.implicit import Implicit
from sparse_ho.criterion import Logistic
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search

n_samples = 100
n_features = 1000
X_train, y_train = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features, n_informative=50,
    random_state=110, flip_y=0.1, n_redundant=0)
X_train_s = csc_matrix(X_train)


X_val, y_val = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features, n_informative=50,
    random_state=122, flip_y=0.1, n_redundant=0)

X_val_s = csc_matrix(X_val)

y_train[y_train == 0.0] = -1.0
y_val[y_val == 0.0] = -1.0

alpha_max = np.max(np.abs(X_train.T @ (- y_train)))
alpha_max /= (2 * n_samples)
alpha = 0.3 * alpha_max
log_alpha = np.log(alpha)
tol = 1e-16

models = [
    SparseLogreg(
        X_train, y_train, max_iter=10000, estimator=None),
    SparseLogreg(
        X_train_s, y_train, max_iter=10000, estimator=None)
]

estimator = LogisticRegression(
    penalty="l1", tol=1e-12, fit_intercept=False, max_iter=100000,
    solver="saga")

models_custom = [
    SparseLogreg(
        X_train, y_train, max_iter=10000, estimator=estimator),
    SparseLogreg(
        X_train_s, y_train, max_iter=10000, estimator=estimator)
]


def get_v(mask, dense):
    return 2 * (X_val[:, mask].T @ (
        X_val[:, mask] @ dense - y_val)) / X_val.shape[0]


@pytest.mark.parametrize('model', models)
def test_beta_jac(model):
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X_train, y_train, log_alpha, tol=tol,
        model=model, compute_jac=True, max_iter=1000)

    clf = LogisticRegression(penalty="l1", tol=1e-12, C=(
        1 / (alpha * n_samples)), fit_intercept=False, max_iter=100000,
        solver="saga")
    clf.fit(X_train, y_train)
    supp_sk = clf.coef_ != 0
    dense_sk = clf.coef_[supp_sk]

    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X_train, y_train, log_alpha,
        get_v, tol=tol, model=model, tol_jac=1e-12)

    supp3, dense3, jac3 = get_beta_jac_iterdiff(
        X_train, y_train, log_alpha, tol=tol,
        model=model, compute_jac=True, max_iter=1000)

    supp4, dense4, jac4 = get_beta_jac_fast_iterdiff(
        X_train_s, y_train, log_alpha,
        get_v, tol=tol, model=model, tol_jac=1e-12)

    assert np.all(supp1 == supp_sk)
    assert np.allclose(dense1, dense_sk, atol=1e-4)

    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense2)
    assert np.allclose(jac1, jac2, atol=1e-4)

    assert np.all(supp2 == supp3)
    assert np.allclose(dense2, dense3)
    assert np.allclose(jac2, jac3, atol=1e-4)

    assert np.all(supp3 == supp4)
    assert np.allclose(dense3, dense4)
    assert np.allclose(jac3, jac4, atol=1e-4)


@pytest.mark.parametrize(('model', 'model_custom'), (models, models_custom))
def test_beta_jac_custom_solver(model, model_custom):
    supp, dense, jac = get_beta_jac_fast_iterdiff(
        X_train, y_train, log_alpha,
        get_v, tol=tol, model=model, tol_jac=1e-12)

    supp_custom, dense_custom, jac_custom = get_beta_jac_fast_iterdiff(
        X_train, y_train, log_alpha, get_v, tol=tol, model=model_custom,
        tol_jac=1e-12)

    assert np.all(supp == supp_custom)
    assert np.allclose(dense, dense_custom)
    assert np.allclose(jac, jac_custom)


@pytest.mark.parametrize('model', models)
def test_val_grad(model):
    criterion = Logistic(X_val, y_val, model)
    algo = Forward()
    val_fwd, grad_fwd = criterion.get_val_grad(
        log_alpha, algo.get_beta_jac_v, tol=tol)

    criterion = Logistic(X_val, y_val, model)
    algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=5000)
    val_imp_fwd, grad_imp_fwd = criterion.get_val_grad(
        log_alpha, algo.get_beta_jac_v, tol=tol)

    criterion = Logistic(X_val, y_val, model)
    algo = Implicit()
    val_imp, grad_imp = criterion.get_val_grad(
        log_alpha, algo.get_beta_jac_v, tol=tol)

    assert np.allclose(val_fwd, val_imp_fwd, atol=1e-4)
    assert np.allclose(grad_fwd, grad_imp_fwd, atol=1e-4)
    assert np.allclose(val_imp_fwd, val_imp, atol=1e-4)

    # for the implcit the conjugate grad does not converge
    # hence the rtol=1e-2
    assert np.allclose(grad_imp_fwd, grad_imp, rtol=1e-2)


@pytest.mark.parametrize(('model', 'model_custom'), (models, models_custom))
def test_val_grad_custom(model, model_custom):
    criterion = Logistic(X_val, y_val, model)
    algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=5000)
    val, grad = criterion.get_val_grad(
        log_alpha, algo.get_beta_jac_v, tol=tol)

    criterion = Logistic(X_val, y_val, model_custom)
    algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=5000)
    val_custom, grad_custom = criterion.get_val_grad(
        log_alpha, algo.get_beta_jac_v, tol=tol)

    assert np.allclose(val, val_custom)
    assert np.allclose(grad, grad_custom)


@pytest.mark.parametrize('model', models)
@pytest.mark.parametrize('crit', ['cv'])
def test_grad_search(model, crit):
    """check that the paths are the same in the line search"""
    n_outer = 2

    criterion = Logistic(X_val, y_val, model)
    monitor1 = Monitor()
    algo = Forward()
    grad_search(algo, criterion, log_alpha, monitor1, n_outer=n_outer,
                tol=tol)

    criterion = Logistic(X_val, y_val, model)
    monitor2 = Monitor()
    algo = Implicit()
    grad_search(algo, criterion, log_alpha, monitor2, n_outer=n_outer,
                tol=tol)

    criterion = Logistic(X_val, y_val, model)
    monitor3 = Monitor()
    algo = ImplicitForward(tol_jac=tol, n_iter_jac=5000)
    grad_search(algo, criterion, log_alpha, monitor3, n_outer=n_outer,
                tol=tol)

    assert np.allclose(
        np.array(monitor1.log_alphas), np.array(monitor3.log_alphas))
    assert np.allclose(
        np.array(monitor1.grads), np.array(monitor3.grads), atol=1e-4)
    assert np.allclose(
        np.array(monitor1.objs), np.array(monitor3.objs))
    assert not np.allclose(
        np.array(monitor1.times), np.array(monitor3.times))


@pytest.mark.parametrize(('model', 'model_custom'), (models, models_custom))
@pytest.mark.parametrize('crit', ['cv'])
def test_grad_search_custom(model, model_custom, crit):
    """check that the paths are the same in the line search"""
    n_outer = 5

    criterion = Logistic(X_val, y_val, model)
    monitor = Monitor()
    algo = ImplicitForward(criterion, tol_jac=tol, n_iter_jac=5000)
    grad_search(algo, log_alpha, monitor, n_outer=n_outer, tol=tol)

    criterion = Logistic(X_val, y_val, model_custom)
    monitor_custom = Monitor()
    algo = ImplicitForward(criterion, tol_jac=tol, n_iter_jac=5000)
    grad_search(algo, log_alpha, monitor_custom, n_outer=n_outer, tol=tol)

    assert np.allclose(
        np.array(monitor.log_alphas), np.array(monitor_custom.log_alphas))
    assert np.allclose(
        np.array(monitor.grads), np.array(monitor_custom.grads), atol=1e-4)
    assert np.allclose(
        np.array(monitor.objs), np.array(monitor_custom.objs))
    assert not np.allclose(
        np.array(monitor.times), np.array(monitor_custom.times))


if __name__ == '__main__':
    for model in models:
        test_beta_jac(model)
        test_grad_search(model, 'cv')
        test_val_grad(model)
