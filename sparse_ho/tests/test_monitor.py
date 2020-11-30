import numpy as np
from numpy.linalg import norm
from sklearn.model_selection import train_test_split
from sklearn import linear_model

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho import grad_search
from sparse_ho.datasets import get_synt_data
from sparse_ho.optimizers import LineSearch


X_, y_, beta_star = get_synt_data(
    n_samples=200, n_features=70, n_times=1, SNR=5)[:3]

X, X_unseen, y, y_unseen = train_test_split(X_, y_)

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train]))) / len(idx_train)
log_alpha0 = np.log(alpha_max / 10)

tol = 1e-7
max_iter = 1e5


estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=max_iter, warm_start=True)


objs = []


def callback(val, grad, mask, dense, log_alpha):
    beta = np.zeros(len(mask))
    beta[mask] = dense
    objs.append(
        norm(X[np.ix_(idx_val, mask)] @ dense - y[idx_val]) ** 2 / len(idx_val))


def test_monitor():
    model = Lasso(estimator=estimator)
    criterion = HeldOutMSE(idx_train, idx_val)
    algo = ImplicitForward()
    monitor = Monitor(callback=callback)
    optimizer = LineSearch(n_outer=10, tol=tol)
    grad_search(algo, criterion, model, optimizer, X, y, log_alpha0, monitor)

    np.testing.assert_allclose(np.array(monitor.objs), np.array(objs))


if __name__ == '__main__':
    test_monitor()
