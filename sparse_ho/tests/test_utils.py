import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
import celer
from celer.datasets import make_correlated_data

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho import grad_search
from sparse_ho.optimizers import LineSearch


n_samples, n_features, corr, snr = 200, 70, 0.1, 5

X, y, _ = make_correlated_data(
    n_samples, n_features, corr=corr, snr=snr, random_state=42)

X, _, y, _ = train_test_split(X, y)

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

n_samples = len(y[idx_train])
alpha_max = np.max(
    np.abs(X[idx_train, :].T.dot(y[idx_train]))) / len(idx_train)
alpha0 = alpha_max / 10

tol = 1e-7
max_iter = 100_000


estimator = celer.Lasso(
    fit_intercept=False, max_iter=50, warm_start=True)


objs = []
X_val = X[idx_val]


def callback(val, grad, mask, dense, log_alpha):
    beta = np.zeros(len(mask))
    beta[mask] = dense
    objs.append(
        norm(X_val[:, mask] @ dense - y[idx_val]) ** 2 / len(idx_val))


def test_monitor():
    model = Lasso(estimator=estimator)
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward()
    monitor = Monitor(callback=callback)
    optimizer = LineSearch(n_outer=10, tol=tol)
    grad_search(algo, criterion, model, optimizer, X, y, alpha0, monitor)

    np.testing.assert_allclose(np.array(monitor.objs), np.array(objs))


if __name__ == '__main__':
    test_monitor()
