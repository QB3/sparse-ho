import numpy as np

from libsvmdata.datasets import fetch_libsvm
from sklearn.linear_model import LogisticRegression
# from sklearn.linear_model import LogisticRegression

from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import LogisticMulticlass
from sparse_ho import ImplicitForward
from sparse_ho.optimizers import GradientDescent

from sparse_ho.ho import grad_search, hyperopt_wrapper
from sparse_ho.utils import Monitor
from sparse_ho.datasets.utils_datasets import (
    alpha_max_multiclass, clean_dataset, get_splits)


# load data
n_samples = 1_000
n_features = 1_000
# n_samples = 1_100
# n_features = 3_200
# X, y = fetch_libsvm('sensit')
# X, y = fetch_libsvm('usps')
X, y = fetch_libsvm('rcv1_multiclass')
# X, y = fetch_libsvm('sector_scale')
# X, y = fetch_libsvm('sector')
# X, y = fetch_libsvm('smallNORB')
# X, y = fetch_libsvm('mnist')


# clean data and subsample
X, y = clean_dataset(X, y, n_samples, n_features)
idx_train, idx_val, idx_test = get_splits(X, y)
n_samples, n_features = X.shape

algo = ImplicitForward(n_iter_jac=1000)
estimator = LogisticRegression(
    C=1, fit_intercept=False, warm_start=True, max_iter=2000, verbose=False)

model = SparseLogreg(estimator=estimator)
logit_multiclass = LogisticMulticlass(
    idx_train, idx_val, algo, idx_test=idx_test)


alpha_max, n_classes = alpha_max_multiclass(X, y)
tol = 1e-5


n_alphas = 10
p_alphas = np.geomspace(1, 0.001, n_alphas)
p_alphas = np.tile(p_alphas, (n_classes, 1))

print("###################### GRID SEARCH ###################")
monitor_grid = Monitor()
for i in range(n_alphas):
    log_alpha_i = np.log(alpha_max * p_alphas[:, i])
    logit_multiclass.get_val(
        model, X, y, log_alpha_i, None, monitor_grid, tol)

1/0
print("###################### GRAD SEARCH LS ###################")
n_outer = 100
model = SparseLogreg(estimator=estimator)
logit_multiclass = LogisticMulticlass(idx_train, idx_val, idx_test, algo)

monitor = Monitor()
log_alpha0 = np.ones(n_classes) * np.log(0.1 * alpha_max)

idx_min = np.argmin(np.array(monitor_grid.objs))
log_alpha0 = monitor_grid.log_alphas[idx_min]
optimizer = GradientDescent(
    n_outer=n_outer, step_size=None, p_grad_norm=0.1, tol=tol)
grad_search(
    algo, logit_multiclass, model, optimizer, X, y, log_alpha0, monitor)


print("###################### USE HYPEROPT ###################")
log_alpha_max = np.log(alpha_max)
log_alpha_min = np.log(alpha_max / 10_000)
monitor_hyp = Monitor()
hyperopt_wrapper(
    algo, logit_multiclass, model, X, y, log_alpha_min, log_alpha_max,
    monitor_hyp, tol=tol, size_space=n_classes, max_evals=10)
