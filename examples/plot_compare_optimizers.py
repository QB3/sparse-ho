"""
========================
Compare outer optimizers
========================

This example shows how to perform hyperparameter optimization
for sparse logistic regression using a held-out test set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#          Mathurin Massias
#
# License: BSD (3-clause)


import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from libsvmdata.datasets import fetch_libsvm

from sparse_ho import ImplicitForward
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import HeldOutLogistic
from sparse_ho.utils_plot import discrete_cmap
from sparse_ho.grid_search import grid_search
from sparse_ho.optimizers import LineSearch, GradientDescent, Adam


print(__doc__)

dataset = 'rcv1.binary'
# dataset = 'simu'

if dataset != 'simu':
    X, y = fetch_libsvm(dataset)
    X = X[:, :100]
else:
    X, y = make_classification(
        n_samples=100, n_features=1_000, random_state=42, flip_y=0.02)


n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T @ y[idx_train]))

alpha_max /= 2 * len(idx_train)
alpha_max = alpha_max
alpha_min = alpha_max / 100
max_iter = 100

tol = 1e-8

n_alphas = 30
p_alphas = np.geomspace(1, 0.0001, n_alphas)
alphas = alpha_max * p_alphas

##############################################################################
# Grid-search
# -----------

estimator = LogisticRegression(
    penalty='l1', fit_intercept=False, max_iter=max_iter)
model = SparseLogreg(estimator=estimator)
criterion = HeldOutLogistic(idx_train, idx_val)
monitor_grid = Monitor()
grid_search(
    criterion, model, X, y, alpha_min, alpha_max,
    monitor_grid, alphas=alphas, tol=tol)
objs = np.array(monitor_grid.objs)


##############################################################################
# Grad-search
# -----------
optimizer_names = ['line-search', 'gradient-descent', 'adam']
optimizers = {
    'line-search': LineSearch(n_outer=10, tol=tol),
    'gradient-descent': GradientDescent(n_outer=10, step_size=100),
    'adam': Adam(n_outer=10, lr=0.11)}

monitors = {}
alpha0 = alpha_max / 10  # starting point

for optimizer_name in optimizer_names:
    estimator = LogisticRegression(
        penalty='l1', fit_intercept=False, solver='saga', tol=tol)
    model = SparseLogreg(estimator=estimator)
    criterion = HeldOutLogistic(idx_train, idx_val)

    monitor_grad = Monitor()
    algo = ImplicitForward(tol_jac=tol, n_iter_jac=1000)

    optimizer = optimizers[optimizer_name]
    grad_search(
        algo, criterion, model, optimizer, X, y, alpha0,
        monitor_grad)
    monitors[optimizer_name] = monitor_grad


current_palette = sns.color_palette("colorblind")
dict_colors = {
    'line-search': 'Greens',
    'gradient-descent': 'Purples',
    'adam': 'Reds'}


fig, ax = plt.subplots(figsize=(8, 3))
ax.plot(alphas / alphas[0], objs, color=current_palette[0])
ax.plot(
    alphas / alphas[0], objs, 'bo',
    label='0-order method (grid-search)', color=current_palette[1])

for optimizer_name in optimizer_names:
    monitor = monitors[optimizer_name]
    p_alphas_grad = np.array(monitor.alphas) / alpha_max
    objs_grad = np.array(monitor.objs)
    cmap = discrete_cmap(len(p_alphas_grad), dict_colors[optimizer_name])
    ax.scatter(
        p_alphas_grad, objs_grad, label=optimizer_name,
        marker='X', color=cmap(np.linspace(0, 1, 10)), zorder=10)

ax.set_xlabel(r"$\lambda / \lambda_{\max}$")
ax.set_ylabel(
    r"$ \sum_i^n \log \left ( 1 + e^{-y_i^{\rm{val}} X_i^{\rm{val}} "
    r"\hat \beta^{(\lambda)} } \right ) $")

ax.set_xscale("log")
plt.tick_params(width=5)
plt.legend()
plt.tight_layout()
plt.show(block=False)
