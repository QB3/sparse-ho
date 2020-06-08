import numpy as np
import pytest
from sklearn import datasets
from sklearn.svm import LinearSVC

from sparse_ho.models import SVM
from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff


n_samples = 1000
n_features = 100
X_train, y_train = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features, n_informative=50,
    random_state=110, flip_y=0.1, n_redundant=0)


X_val, y_val = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features, n_informative=50,
    random_state=122, flip_y=0.1, n_redundant=0)


y_train[y_train == 0.0] = -1.0
y_val[y_val == 0.0] = -1.0


C = 1e-3
log_C = np.log(C)
tol = 1e-16

models = [
    SVM(
        X_train, y_train, log_C, max_iter=10000, tol=tol)
]


def get_v(mask, dense):
    return 2 * (X_val[:, mask].T @ (
        X_val[:, mask] @ dense - y_val)) / X_val.shape[0]


@pytest.mark.parametrize('model', models)
def test_beta_jac(model):
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X_train, y_train, log_C, tol=tol,
        model=model, compute_jac=True, max_iter=1000)

    primal = np.sum(y_train[supp1] * dense1 * X_train[supp1, :].T, axis=1)
    clf = LinearSVC(
        loss="hinge", fit_intercept=False, C=C, tol=tol, max_iter=100000)

    clf.fit(X_train, y_train)

    supp2, dense2, jac2 = get_beta_jac_fast_iterdiff(
        X_train, y_train, log_C, None, None,
        get_v, tol=tol, model=model, tol_jac=1e-16)

    assert np.allclose(primal, clf.coef_)

    assert np.all(supp1 == supp2)
    assert np.allclose(dense1, dense2)
    assert np.allclose(jac1, jac2, atol=1e-4)


if __name__ == '__main__':
    for model in models:
        test_beta_jac(model)
