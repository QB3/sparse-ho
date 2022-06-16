import numpy as np
import sklearn.linear_model

from libsvmdata.datasets import fetch_libsvm

from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import LogisticMulticlass
from sparse_ho import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.datasets.utils_datasets import (
    alpha_max_multiclass, clean_dataset)


# load data
n_samples = 1000
n_features = 10
X, y = fetch_libsvm('mnist')
my_bool = np.logical_or(np.logical_or(y == 0, y == 1), y == 2)

X = X[my_bool, :]
y = y[my_bool]
# clean data and subsample
X, y = clean_dataset(X, y, n_samples, n_features)
idx_train = np.arange(len(y) // 2)
idx_val = np.arange(len(y) // 2, len(y))

alpha_max, n_classes = alpha_max_multiclass(X, y)
tol = 1e-8

n_classes = np.unique(y).shape[0]

max_iter = 10000
algo = ImplicitForward(n_iter_jac=1000)
estimator = sklearn.linear_model.LogisticRegression(
    solver='saga', penalty='l1', max_iter=max_iter,
    random_state=42, fit_intercept=False, warm_start=True)

model = SparseLogreg(estimator=estimator)
logit_multiclass = LogisticMulticlass(
    idx_train=idx_train, idx_val=idx_val, algo=algo)

n_alphas = 10
p_alphas = np.geomspace(1, 0.1, n_alphas)
p_alphas = np.tile(p_alphas, (n_classes, 1))


def test_our_vs_sklearn():
    monitor_grid = Monitor()
    monitor_grid_sk = Monitor()
    for i in range(n_alphas):
        # one versus all (ovr) logreg from scikit learn
        p_alpha = p_alphas[:, i]
        lr = sklearn.linear_model.LogisticRegression(
            solver='saga', multi_class='ovr', penalty='l1', max_iter=max_iter,
            random_state=42, fit_intercept=False, warm_start=True,
            C=1 / (alpha_max * p_alpha[0] * len(idx_train)), tol=tol)
        lr.fit(X[idx_train, :], y[idx_train])
        y_pred_val = lr.predict(X[idx_val, :])
        accuracy_val = sklearn.metrics.accuracy_score(y_pred_val, y[idx_val])
        print("accuracy validation (scikit) %f " % accuracy_val)

        monitor_grid_sk(None, None, acc_val=accuracy_val)
        log_alpha_i = np.log(alpha_max * p_alpha)
        # our one verus all
        val, grad = logit_multiclass.get_val_grad(
            model, X, y, log_alpha_i, None, monitor_grid, tol)
        print("accuracy validation (our) %f " % monitor_grid.acc_vals[-1])

    np.testing.assert_allclose(
        np.array(monitor_grid.acc_vals), np.array(monitor_grid_sk.acc_vals))


if __name__ == '__main__':
    test_our_vs_sklearn()
