"""
=============================
Lasso with Cross-validation
=============================

This example shows how to perform hyperparameter optimization
for a Lasso using a full cross-validation score.
"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search

print(__doc__)

# dataset = 'rcv1'
dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1_train')
else:
    X, y = make_regression(
        n_samples=500, n_features=1000, noise=40,
        random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting path computation...")
n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples

n_alphas = 10
p_alphas = np.geomspace(1, 0.001, n_alphas)
alphas = alpha_max * p_alphas

tol = 1e-8

max_iter = 1e5

##############################################################################
# Cross-validation with scikit-learn
# ----------------------------------
print('scikit started')

t0 = time.time()
reg = LassoCV(
    cv=kf, verbose=True, tol=tol, fit_intercept=False,
    alphas=alphas, max_iter=max_iter).fit(X, y)
reg.score(X, y)
t_sk = time.time() - t0

print('scikit finished')

##############################################################################
# Now do the hyperparameter optimization with implicit differentiation
# --------------------------------------------------------------------

estimator = sklearn.linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True, tol=tol)

print('sparse-ho started')

t0 = time.time()
Model = Lasso
Criterion = HeldOutMSE
log_alpha0 = np.log(alpha_max / 10)
monitor_grad = Monitor()
criterion = CrossVal(X, y, Model, cv=kf, estimator=estimator)
algo = ImplicitForward()
grad_search(
    algo, criterion, np.log(alpha_max / 10), monitor_grad, n_outer=10, tol=tol)

t_grad_search = time.time() - t0

print('sparse-ho finished')

##############################################################################
# Plot results
# ------------
objs = reg.mse_path_.mean(axis=1)

p_alphas_grad = np.exp(np.array(monitor_grad.log_alphas)) / alpha_max
objs_grad = np.array(monitor_grad.objs)


print("Time to compute CV for scikit-learn: %.2f" % t_sk)
print("Time to compute CV for sparse-ho: %.2f" % t_grad_search)

print('Minimum objective grid-search %.5f' % objs.min())
print('Minimum objective grad-search %.5f' % objs_grad.min())


current_palette = sns.color_palette("colorblind")

fig = plt.figure(figsize=(5, 3))
plt.semilogx(
    p_alphas, objs, color=current_palette[0])
plt.semilogx(
    p_alphas, objs, 'bo', label='0-order method (grid-search)',
    color=current_palette[1])
plt.semilogx(
    p_alphas_grad, objs_grad, 'bX', label='1-st order method',
    color=current_palette[2])
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel("Cross-validation loss")
axes = plt.gca()
plt.tick_params(width=5)
plt.legend()
plt.tight_layout()
plt.show(block=False)
