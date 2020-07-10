"""
===========================
Sparse logistic regression
===========================

This example shows how to perform hyperparameter optimisation
for sparse logistic regression using a held-out test set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import Logistic
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.forward import Forward
from sparse_ho.grid_search import grid_search
from sparse_ho.datasets.real import get_rcv1
from sparse_ho.datasets.real import get_real_sim

print(__doc__)

dataset = 'rcv1'
# dataset = 'simu'

if dataset == 'rcv1':
    X_train, X_val, X_test, y_train, y_val, y_test = get_rcv1()
else:
    X_train, X_val, X_test, y_train, y_val, y_test = get_real_sim()


n_samples, n_features = X_train.shape

alpha_max = np.max(np.abs(X_train.T @ y_train))
alpha_max /= 4 * n_samples
log_alpha_max = np.log(alpha_max)
log_alpha_min = np.log(alpha_max / 1000)
maxit = 1000

log_alpha0 = np.log(0.1 * alpha_max)
tol = 1e-7
use_sk = True
# use_sk = False

n_alphas = 10
p_alphas = np.geomspace(1, 0.0001, n_alphas)
alphas = alpha_max * p_alphas
log_alphas = np.log(alphas)

##############################################################################
# Grid-search
# -----------
model = SparseLogreg(X_train, y_train, log_alpha0, max_iter=100)
criterion = Logistic(X_val, y_val, model)
algo_grid = Forward(criterion, use_sk=use_sk)
monitor_grid = Monitor()
grid_search(
    algo_grid, log_alpha_min, log_alpha_max, monitor_grid,
    log_alphas=log_alphas, tol=tol)
objs = np.array(monitor_grid.objs)


##############################################################################
# Grad-search
# -----------
model = SparseLogreg(X_train, y_train, log_alpha0, max_iter=100, tol=tol)
criterion = Logistic(X_val, y_val, model)
monitor_grad = Monitor()
algo = ImplicitForward(criterion, tol_jac=tol, n_iter_jac=100, use_sk=use_sk)
grad_search(algo, log_alpha0, monitor_grad, n_outer=10, tol=tol)
objs_grad = np.array(monitor_grad.objs)


p_alphas_grad = np.exp(np.array(monitor_grad.log_alphas)) / alpha_max

objs_grad = np.array(monitor_grad.objs)

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
    r"$ \sum_i^n \log \left ( 1 + e^{-y_i^{\rm{val}} X_i^{\rm{val}} "
    r"\hat \beta^{(\lambda)} } \right ) $")

axes = plt.gca()
axes.set_ylim([0, 1])
plt.tick_params(width=5)
plt.legend(loc=1)
plt.tight_layout()
plt.show(block=False)
