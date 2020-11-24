"""
============================
Lasso with held-out test set
============================

This example shows how to perform hyperparameter optimization
for a Lasso using a held-out validation set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search
from sparse_ho.grid_search import grid_search
from sklearn.datasets import make_regression

from libsvmdata.datasets import fetch_libsvm


print(__doc__)

dataset = 'rcv1'
# dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1_train')
else:
    X, y = make_regression(n_samples=1000, n_features=1000, noise=40)

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train]))) / len(idx_train)
log_alpha0 = np.log(alpha_max / 10)

n_alphas = 10
p_alphas = np.geomspace(1, 0.0001, n_alphas)
alphas = alpha_max * p_alphas
log_alphas = np.log(alphas)

tol = 1e-7
max_iter = 1e5

##############################################################################
# Grid-search with scikit-learn
# -----------------------------

estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)

print('scikit-learn started')

t0 = time.time()
model = Lasso(estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = Forward()
monitor_grid_sk = Monitor()
grid_search(
    algo, criterion, model, X, y, None, None, monitor_grid_sk, log_alphas=log_alphas, tol=tol)
objs = np.array(monitor_grid_sk.objs)
t_sk = time.time() - t0

print('scikit-learn finished')


##############################################################################
# Grad-search with sparse-ho
# --------------------------

print('sparse-ho started')

t0 = time.time()
model = Lasso(estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward(criterion)
monitor_grad = Monitor()
grad_search(
    algo, criterion, model, X, y, np.log(alpha_max / 10), monitor_grad,
    n_outer=10, tol=tol)

t_grad_search = time.time() - t0

print('sparse-ho finished')

##############################################################################
# Plot results
# ------------

p_alphas_grad = np.exp(np.array(monitor_grad.log_alphas)) / alpha_max

objs_grad = np.array(monitor_grad.objs)

print('sparse-ho finished')
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
plt.ylabel(
    r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$")
plt.tick_params(width=5)
plt.legend()
plt.tight_layout()
plt.show(block=False)
