"""
======================
Grad Search CV
======================

...

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import time
import numpy as np
from numpy.linalg import norm

# from sklearn.datasets import make_regression
from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sparse_ho.models import Lasso
from sparse_ho.criterion import CV
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.grad_search_CV import grad_search_CV
from sparse_ho.datasets.real import load_libsvm

print(__doc__)

X, y = load_libsvm('rcv1train')
# X, y = make_regression(
#     n_samples=2000, n_features=10000)

random_state = 0
cv = 2

print("Starting path computation...")
n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples

n_alphas = 100
alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)

tol = 1e-8

print('scikit started')

t0 = time.time()
reg = LassoCV(
    cv=cv, random_state=random_state, verbose=True, tol=tol, fit_intercept=False, alphas=alphas).fit(X, y)
reg.score(X, y)
t_sk = time.time() - t0

print('scikit finished')


print('sparse-ho started')

t0 = time.time()
Model = Lasso
Criterion = CV
Algo = ImplicitForward
log_alpha0 = np.log(alpha_max/10)
monitor = Monitor()
grad_search_CV(
    X, y, Model, Criterion, Algo, log_alpha0, monitor, n_outer=30,
    verbose=True, cv=cv, random_state=0, test_size=0.33,
    tolerance_decrease='constant', tol=tol,
    t_max=1000)
t_grad_search = time.time() - t0

print('sparse-ho finished')
print("Time to compute CV for scikit-learn: %.2f" % t_sk)
print("Time to compute CV for sparse-ho: %.2f" % t_grad_search)

clf = linear_model.Lasso(alpha=reg.alpha_)
clf.fit(X, y)
norm(X @ clf.coef_ - y) ** 2 / (2 * n_samples) + reg.alpha_ * norm(clf.coef_, ord=1)

clf2 = linear_model.Lasso(alpha=np.exp(monitor.log_alphas[-1]))
clf2.fit(X, y)

norm(X @ clf2.coef_ - y) ** 2 / (2 * n_samples) + np.exp(monitor.log_alphas[-1]) * norm(clf2.coef_, ord=1)
