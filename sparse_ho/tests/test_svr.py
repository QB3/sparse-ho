import numpy as np
from sklearn.svm import LinearSVR
from sparse_ho.models import SVR
from sparse_ho.forward import get_beta_jac_iterdiff
from sklearn.datasets import make_regression
n_samples = 100
n_features = 100
n_active = 100
SNR = 5.0
tol = 1e-12
C = 0.1
log_C = np.log(C)
epsilon = 0.05
log_epsilon = np.log(epsilon)

X_train, y_train, beta_star = make_regression(
    shuffle=False, random_state=10, n_samples=n_samples, n_features=n_features,
    n_informative=n_features, n_targets=1, coef=True)

model = SVR(X_train, y_train, log_C, log_epsilon, max_iter=10000, tol=tol)


def test_beta_jac():
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X_train, y_train, np.array([log_C, log_epsilon]), tol=tol,
        model=model, compute_jac=True, max_iter=10000)

    dual = np.zeros(2 * n_samples)
    dual[supp1] = dense1
    primal = X_train.T @ (dual[0:n_samples] - dual[n_samples:(2 * n_samples)])
    clf = LinearSVR(
        epsilon=epsilon, fit_intercept=False, C=C, tol=tol, max_iter=100000)
    clf.fit(X_train, y_train)

    assert np.allclose(primal, clf.coef_)


if __name__ == '__main__':
    test_beta_jac(model)
