import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import scipy

from libsvmdata.datasets import fetch_libsvm
from celer import LogisticRegression
# from sklearn.linear_model import LogisticRegression

from sparse_ho.criterion import LogisticMulticlass
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.ho_dirty import (
    grad_search, adam_search, grad_search_backtrack_dirty,
    grad_search_backtracking_cd_dirty2, grad_search_scipy, brent_cd)

from sparse_ho.ho import grad_search_wolfe_cd_dirty
from sparse_ho.utils import Monitor
from sparse_ho.utils_datasets import get_alpha_max, clean_dataset

# load data
n_samples = 1_000
n_features = 1_000
# n_samples = 1_100
# n_features = 3_200
# X, y = fetch_libsvm('sensit')
# X, y = fetch_libsvm('usps')
# X, y = fetch_libsvm('rcv1_multiclass', normalize=True)
# X, y = fetch_libsvm('sector_scale')
# X, y = fetch_libsvm('sector')
# X, y = fetch_libsvm('smallNORB')
X, y = fetch_libsvm('mnist')

X, y = clean_dataset(X, y, n_samples, n_features)
n_samples, n_features = X.shape

algo = ImplicitForward(None, n_iter_jac=1000)
estimator = LogisticRegression(
    C=1, fit_intercept=False, warm_start=True, max_iter=200, verbose=False)
# C=1, fit_intercept=False, warm_start=True, verbose=True)
# estimator = LogisticRegression(
#     penalty='l1', C=1, fit_intercept=False, warm_start=True, solver='saga')
logit_multiclass = LogisticMulticlass(X, y, algo, estimator)

alpha_max, n_classes = get_alpha_max(X, y)

monitor = Monitor()
tol = 1e-8
method = 'random'

log_alpha_max = np.log(alpha_max)

n_alphas = 10
p_alphas = np.geomspace(1, 0.001, n_alphas)
p_alphas = np.tile(p_alphas, (n_classes, 1))

monitor_grid = Monitor()

for i in range(n_alphas):
    val, grad = logit_multiclass.get_val_grad(
        np.log(alpha_max * p_alphas[:, i]), monitor_grid)
    print("%i / %i  || crosss entropy %f  || accuracy val %f  || accuracy test %f" % (
        i, n_alphas, val, monitor_grid.acc_vals[-1], monitor_grid.acc_tests[-1]))

print("min cross entropy grid-search %f " % np.array(np.min(monitor_grid.objs)))
print("max accuracy grid-search %f " % np.array(np.max(monitor_grid.acc_vals)))

tol = 1e-8

# # 1 / 0

# print("###################### ADAM ###################")
# monitor_adam = Monitor()
# log_alpha0 = np.ones(n_classes) * np.log(0.1 * alpha_max)
# lr = 0.01
# beta_2 = 0.999
# n_outer = 2000
# # idx_min = np.argmin(np.array(monitor_grid.objs))
# # log_alpha0 = monitor_grid.log_alphas[idx_min]
# adam_search(
#     logit_multiclass, log_alpha0, monitor_adam, n_outer=n_outer, verbose=1, epsilon=1e-8, lr=lr, beta_2=beta_2, tol=tol)

# 1 / 0

# idx_min = np.argmin(np.array(monitor_adam.objs))
# log_alpha0 = monitor_adam.log_alphas[-1]

# n_outer = 1_000
# # log_alpha0 = monitor_adam.log_alphas[-1]
# adam_search(
#     logit_multiclass, log_alpha0, monitor_adam, n_outer=n_outer, verbose=1, epsilon=1e-8, lr=lr/10)

# 1/0

# print("###################### GRAD SEARCH LS ###################")
# n_outer = 10
# monitor = Monitor()
# # log_alpha0 = np.ones(n_classes) * np.log(0.1 * alpha_max)

# idx_min = np.argmin(np.array(monitor_grid.objs))
# log_alpha0 = monitor_grid.log_alphas[idx_min]
# grad_search(
#     logit_multiclass, log_alpha0, monitor,
#     n_outer=n_outer, tol=1e-7)
# grad_search_scipy(
#     logit_multiclass, log_alpha0, monitor, n_outer=n_outer, verbose=1, tol=1e-7)


# idx_min = np.argmin(np.array(monitor.objs))
# log_alpha0 = monitor.log_alphas[idx_min]

# 1 / 0
# print("###################### GRAD SEARCH backtrack ###################")

# n_outer = 4
# monitor = Monitor()

# idx_min = np.argmin(np.array(monitor_grid.objs))
# log_alpha0 = monitor_grid.log_alphas[idx_min]
# logit_multiclass.init_beta = False
n_outer = 10
log_alpha0 = np.ones(n_classes) * np.log(0.01 * alpha_max)
grad_search_backtracking_cd_dirty2(
    logit_multiclass, log_alpha0, monitor, n_outer=n_outer, tol=tol,
    maxit_ln=20)

# brent_cd(
#     logit_multiclass, log_alpha0, monitor, n_outer=n_outer, tol=tol)

# idx_min = np.argmin(np.array(monitor.objs))
# log_alpha0 = monitor.log_alphas[idx_min]
1 / 0



# print("###################### GRAD SEARCH backtracking LS ###################")
# n_outer = 10
# monitor = Monitor()
# log_alpha0 = np.ones(n_classes) * np.log(0.1 * alpha_max)

# # idx_min = np.argmin(np.array(monitor_grid.objs))
# # log_alpha0 = monitor_grid.log_alphas[idx_min]
# grad_search_backtrack_dirty(
#     logit_multiclass, log_alpha0, monitor, n_outer=n_outer, tol=1e-7)

# idx_min = np.argmin(np.array(monitor.objs))
# log_alpha0 = monitor.log_alphas[idx_min]
