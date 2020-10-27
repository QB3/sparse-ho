import numpy as np
from numpy.linalg import norm
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

from celer.datasets import fetch_libsvm
from celer import LogisticRegression

from sparse_ho.criterion import LogisticMulticlass
from sparse_ho.implicit_forward import ImplicitForward

# load data
n_samples = 1000
n_features = 1000
# X, y = fetch_libsvm('smallNORB')
X, y = fetch_libsvm('mnist')
# X, y = fetch_libsvm('rcv1_multiclass')
np.random.seed(0)
idx = np.random.choice(X.shape[0], n_samples, replace=False)
feats = np.random.choice(
    X.shape[1], min(n_features, X.shape[1]), replace=False)
X = X[idx, :]
X = X[:, feats]
y = y[idx]

ypd = pd.DataFrame(y)
bool_rm = ypd.groupby(0)[0].transform(len) > 1
ypd = ypd[bool_rm]
X = X[bool_rm.to_numpy(), :]
y = y[bool_rm.to_numpy()]

n_samples, n_features = X.shape

algo = ImplicitForward(None)
estimator = LogisticRegression(
    C=1, fit_intercept=False, warm_start=True, verbose=True)
logit_multiclass = LogisticMulticlass(X, y, algo, estimator)


alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha_max /= 2
p_alphas = np.geomspace(0.1, 0.001, num=10)
values = np.zeros_like(p_alphas)
for i, p_alpha in enumerate(p_alphas):
    print(i)
    value = logit_multiclass.get_val_grad(np.log(alpha_max * p_alpha))
    values[i] = value


print(values)
