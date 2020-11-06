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
from sparse_ho.ho_dirty import grad_search, adam_search
# , lbfgs
# from sparse_ho.ho import grad_search_wolfe_dirty
from sparse_ho.utils import Monitor


# load data
n_samples = 1_100
n_features = 3_200
# X, y = fetch_libsvm('smallNORB')
X, y = fetch_libsvm('mnist')
# y[y != 1] = 2
# X, y = fetch_libsvm('sector')
# X, y = fetch_libsvm('news20_multiclass')
# X, y = fetch_libsvm('aloi')
# X, y = fetch_libsvm('rcv1_multiclass')
bool_123 = np.logical_or(np.logical_or(y == 1, y == 2), y == 3)
y = y[bool_123]
X = X[bool_123, :]

bool3 = y == 3
bool3[:(bool3.shape[0] * 8 // 10)] = False

bool_123 = np.logical_or(np.logical_or(y == 1, y == 2), bool3)
y = y[bool_123]
X = X[bool_123, :]


np.random.seed(0)
idx = np.random.choice(X.shape[0], min(n_samples, X.shape[0]), replace=False)
feats = np.random.choice(
    X.shape[1], min(n_features, X.shape[1]), replace=False)
X = X[idx, :]
X = X[:, feats]
y = y[idx]

bool_to_keep = scipy.sparse.linalg.norm(X, axis=0) != 0
X = X[:, bool_to_keep]
bool_to_keep = scipy.sparse.linalg.norm(X, axis=1) != 0
X = X[bool_to_keep, :]
y = y[bool_to_keep]

ypd = pd.DataFrame(y)
bool_to_keep = ypd.groupby(0)[0].transform(len) > 1
ypd = ypd[bool_to_keep]
X = X[bool_to_keep.to_numpy(), :]
y = y[bool_to_keep.to_numpy()]
enc = OneHotEncoder(sparse=False)
one_hot_code = enc.fit_transform(ypd)
n_classes = one_hot_code.shape[1]

bool_to_keep = scipy.sparse.linalg.norm(X, axis=0) != 0
X = X[:, bool_to_keep]
bool_to_keep = scipy.sparse.linalg.norm(X, axis=1) != 0
X = X[bool_to_keep, :]
y = y[bool_to_keep]


alpha_max = np.infty
for k in range(n_classes):
    alpha_max = min(alpha_max, norm(
        X.T @ (2 * one_hot_code[:, k] - 1), ord=np.inf) / (2 * n_samples))
    if alpha_max == 0:
        1 / 0

if alpha_max == 0:
    1 / 0

n_samples, n_features = X.shape

algo = ImplicitForward(None, n_iter_jac=200)
estimator = LogisticRegression(
    C=1, fit_intercept=False, warm_start=True, max_iter=50, verbose=False)
# C=1, fit_intercept=False, warm_start=True, verbose=True)
# estimator = LogisticRegression(
#     penalty='l1', C=1, fit_intercept=False, warm_start=True, solver='saga')
logit_multiclass = LogisticMulticlass(X, y, algo, estimator)


n_alphas = 20
p_alphas = np.geomspace(0.8, 0.0001, n_alphas)
p_alphas = np.tile(p_alphas, (n_classes, 1))

monitor_grid = Monitor()

for i in range(n_alphas):
    val, grad = logit_multiclass.get_val_grad(
        np.log(alpha_max * p_alphas[:, i]), monitor_grid)
    print("%i / %i  || crosss entropy %f  || accuracy %f" % (
        i, n_alphas, val, monitor_grid.acc_vals[-1]))

print("min cross entropy grid-search %f " % np.array(np.min(monitor_grid.objs)))
print("max accuracy grid-search %f " % np.array(np.max(monitor_grid.acc_vals)))

# monitor_adam = Monitor()
# log_alpha0 = np.ones(n_classes) * np.log(0.1 * alpha_max)

# lr = 0.01
# beta_2 = 0.999
# print("###################### ADAM ###################")
# n_outer = 2000
# adam_search(
#     logit_multiclass, log_alpha0, monitor_adam, n_outer=n_outer, verbose=1, epsilon=1e-8, lr=lr, beta_2=beta_2)


# idx_min = np.argmin(np.array(monitor_adam.objs))
# log_alpha0 = monitor_adam.log_alphas[-1]

# n_outer = 1_000
# # log_alpha0 = monitor_adam.log_alphas[-1]
# adam_search(
#     logit_multiclass, log_alpha0, monitor_adam, n_outer=n_outer, verbose=1, epsilon=1e-8, lr=lr/10)


print("###################### GRAD SEARCH LS ###################")
log_alpha0 = np.ones(n_classes) * np.log(0.1 * alpha_max)


n_outer = 500
monitor = Monitor()

# idx_min = np.argmin(np.array(monitor_grid.objs))
# log_alpha0 = monitor_grid.log_alphas[idx_min]
grad_search(
    logit_multiclass, log_alpha0, monitor, n_outer=n_outer, verbose=1, tol=1e-7)


idx_min = np.argmin(np.array(monitor.objs))
log_alpha0 = monitor.log_alphas[idx_min]
