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
from sparse_ho.ho_dirty import grad_search, lbfgs
from sparse_ho.ho import grad_search_wolfe_dirty
from sparse_ho.utils import Monitor


# load data
n_samples = 1000
n_features = 10000
# X, y = fetch_libsvm('smallNORB')
# X, y = fetch_libsvm('protein')
# X, y = fetch_libsvm('mnist')
# y[y != 1] = 2
# X, y = fetch_libsvm('sector')
# X, y = fetch_libsvm('news20_multiclass')
X, y = fetch_libsvm('rcv1_multiclass')
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
        import ipdb; ipdb.set_trace()

if alpha_max == 0:
    1 / 0

n_samples, n_features = X.shape

algo = ImplicitForward(None, n_iter_jac=1000)
estimator = LogisticRegression(
    C=1, fit_intercept=False, warm_start=True, max_iter=50)
    # C=1, fit_intercept=False, warm_start=True, verbose=True)
# estimator = LogisticRegression(
#     penalty='l1', C=1, fit_intercept=False, warm_start=True, solver='saga')
logit_multiclass = LogisticMulticlass(X, y, algo, estimator)


n_alphas = 10
p_alphas = np.geomspace(0.1, 0.01, n_alphas)
p_alphas = np.tile(p_alphas, (n_classes, 1))

values = np.zeros(n_alphas)
grads = np.zeros((n_classes, n_alphas))

for i in range(n_alphas):
    print(i)
    val, grad = logit_multiclass.get_val_grad(
        np.log(alpha_max * p_alphas[:, i]))
    print(val)
    values[i] = val
    grads[:, i] = grad

print(values)

n_outer = 10

log_alpha0 = np.ones(n_classes) * np.log(0.1 * alpha_max)
monitor = Monitor()
grad_search(
    logit_multiclass, log_alpha0, monitor, n_outer=n_outer, verbose=1, tol=1e-7)
