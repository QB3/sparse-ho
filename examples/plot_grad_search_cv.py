"""
=============================
Example with cross validation
=============================

...

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# from sklearn.datasets import make_regression
# from sklearn import linear_model
from sklearn.linear_model import LassoCV
from sparse_ho.models import Lasso
from sparse_ho.criterion import CV
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.grad_search_CV import grad_search_CV
from sparse_ho.datasets.real import load_libsvm

from sklearn.model_selection import KFold

print(__doc__)

X, y = load_libsvm('rcv1_train')
# X, y = make_regression(
#     n_samples=2000, n_features=1000)

kf = KFold(n_splits=5, shuffle=False)

for train, test in kf.split(X):
    print("%s %s" % (train, test))

print("Starting path computation...")
n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples

n_alphas = 10
p_alphas = np.geomspace(1, 0.0001, n_alphas)
alphas = alpha_max * p_alphas

tol = 1e-8

print('scikit started')

t0 = time.time()
reg = LassoCV(
    cv=kf, verbose=True, tol=tol, fit_intercept=False, alphas=alphas).fit(X, y)
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
    X, y, Model, Criterion, Algo, log_alpha0, monitor, n_outer=10,
    verbose=True, cv=kf, random_state=0, test_size=0.33,
    tolerance_decrease='constant', tol=tol,
    t_max=1000)
t_grad_search = time.time() - t0

print('sparse-ho finished')
print("Time to compute CV for scikit-learn: %.2f" % t_sk)
print("Time to compute CV for sparse-ho: %.2f" % t_grad_search)


objs = reg.mse_path_.mean(axis=1)

p_alphas_grad = np.exp(np.array(monitor.log_alphas)) / alpha_max
objs_grad = np.array(monitor.objs)

current_palette = sns.color_palette("colorblind")


fig = plt.figure()
plt.semilogx(
    p_alphas, objs, color=current_palette[0], linewidth=7.0)
plt.semilogx(
    p_alphas, objs, 'bo', label='0-order method (grid-search)',
    color=current_palette[1], markersize=15)
plt.semilogx(
    p_alphas_grad, objs_grad, 'bX', label='1-st order method',
    color=current_palette[2], markersize=25)
plt.xlabel(r"$\lambda / \lambda_{\max}$", fontsize=28)
plt.ylabel(
    "Cross-validation loss",
    fontsize=28)
axes = plt.gca()
# axes.set_ylim([0, 1])
plt.tick_params(width=5)
plt.legend(fontsize=14, loc=1)
plt.tight_layout()
plt.show(block=False)
