"""
============================
Lasso with held-out test set
============================

This example shows how to perform hyperparameter optimization
for a Lasso using a held-out validation set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#          Mathurin Massias
#
# License: BSD (3-clause)

import time
import celer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from celer.datasets import make_correlated_data
from libsvmdata.datasets import fetch_libsvm

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.utils_plot import discrete_cmap
from sparse_ho.ho import grad_search
from sparse_ho.grid_search import grid_search
from sparse_ho.optimizers import LineSearch


print(__doc__)

dataset = 'rcv1'
# dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1.binary')
else:
    X, y, _ = make_correlated_data(n_samples=1000, n_features=2000,
                                   random_state=0)

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train])))
alpha_max /= len(idx_train)
alpha0 = alpha_max / 5

n_alphas = 10
alphas = np.geomspace(alpha_max, alpha_max/1_000, n_alphas)
tol = 1e-7

##############################################################################
# Grid search with scikit-learn
# -----------------------------

estimator = celer.Lasso(fit_intercept=False, warm_start=True)

print('Grid search started')

t0 = time.time()
model = Lasso(estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
monitor_grid_sk = Monitor()
grid_search(
    criterion, model, X, y, None, None, monitor_grid_sk,
    alphas=alphas, tol=tol)
objs = np.array(monitor_grid_sk.objs)
t_sk = time.time() - t0

print('Grid search finished')


##############################################################################
# Grad-search with sparse-ho
# --------------------------

print('sparse-ho started')

t0 = time.time()
model = Lasso(estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward()
monitor_grad = Monitor()
optimizer = LineSearch(n_outer=10, tol=tol)
grad_search(
    algo, criterion, model, optimizer, X, y, alpha0, monitor_grad)

t_grad_search = time.time() - t0

print('sparse-ho finished')

##############################################################################
# Plot results
# ------------

p_alphas_grad = np.array(monitor_grad.alphas) / alpha_max

objs_grad = np.array(monitor_grad.objs)

print('sparse-ho finished')
print(f"Time for grid search: {t_sk:.2f} s")
print(f"Time for grad search (sparse-ho): {t_grad_search:.2f} s")

print(f'Minimum outer criterion value with grid search: {objs.min():.5f}')
print(f'Minimum outer criterion value with grad search: {objs_grad.min():.5f}')


current_palette = sns.color_palette("colorblind")
cmap = discrete_cmap(len(objs_grad), 'Greens')


fig, ax = plt.subplots(1, 1, figsize=(5, 3))
ax.plot(alphas / alphas[0], objs, color=current_palette[0])
ax.plot(
    alphas / alphas[0], objs, 'bo', label='0-th order method (grid search)',
    color=current_palette[1])
ax.scatter(
    p_alphas_grad, objs_grad, label='1-st order method',  marker='X',
    color=cmap(np.linspace(0, 1, len(objs_grad))), s=40, zorder=40)
ax.set_xlabel(r"$\lambda / \lambda_{\max}$")
ax.set_ylabel(
    r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$")
plt.tick_params(width=5)
plt.legend()
ax.set_xscale("log")
plt.tight_layout()
plt.show(block=False)
