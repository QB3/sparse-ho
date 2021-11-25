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


import time
from libsvmdata.datasets import fetch_libsvm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from celer import LogisticRegression

from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import HeldOutLogistic
from sparse_ho import ImplicitForward
from sparse_ho.grid_search import grid_search
from sparse_ho.optimizers import GradientDescent
from sparse_ho.utils_plot import discrete_cmap

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

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train])))

alpha_max /= 4 * len(idx_train)
alpha_max = alpha_max
alpha_min = alpha_max / 100
max_iter = 100

alpha0 = 0.1 * alpha_max
tol = 1e-8

n_alphas = 20
alphas = np.geomspace(alpha_max,  alpha_max / 1_000, n_alphas)

##############################################################################
# Grid-search
# -----------

print('Grid search started')
t0 = time.time()

estimator = LogisticRegression(
    penalty='l1', fit_intercept=False, max_iter=max_iter)
model = SparseLogreg(estimator=estimator)
criterion = HeldOutLogistic(idx_train, idx_val)
monitor_grid = Monitor()
grid_search(
    criterion, model, X, y, alpha_min, alpha_max,
    monitor_grid, alphas=alphas, tol=tol)
objs = np.array(monitor_grid.objs)

t_grid_search = time.time() - t0

print('scikit finished')
print(f"Time to compute grad search: {t_grid_search:.2f} s")


##############################################################################
# Grad-search
# -----------

print('sparse-ho started')

t0 = time.time()

estimator = LogisticRegression(
    penalty='l1', fit_intercept=False, tol=tol)
model = SparseLogreg(estimator=estimator)
criterion = HeldOutLogistic(idx_train, idx_val)

monitor_grad = Monitor()
algo = ImplicitForward(tol_jac=tol, n_iter_jac=1000)

optimizer = GradientDescent(n_outer=10, tol=tol)
grad_search(
    algo, criterion, model, optimizer, X, y, alpha0,
    monitor_grad)
objs_grad = np.array(monitor_grad.objs)

t_grad_search = time.time() - t0

print('sparse-ho finished')
print(f"Time to compute grad search: {t_grad_search:.2f} s")


p_alphas_grad = np.array(monitor_grad.alphas) / alpha_max

objs_grad = np.array(monitor_grad.objs)

current_palette = sns.color_palette("colorblind")

fig = plt.figure(figsize=(5, 3))
cmap = discrete_cmap(len(p_alphas_grad), "Greens")

plt.plot(alphas / alphas[0], objs, color=current_palette[0])
plt.plot(
    alphas / alphas[0], objs, 'bo',
    label='0-order method (grid-search)', color=current_palette[1])
plt.scatter(
    p_alphas_grad, objs_grad, label='1-st order method',
    marker='X', color=cmap(np.linspace(0, 1, len(objs_grad))), zorder=10)
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel(
    r"$ \sum_i^n \log \left ( 1 + e^{-y_i^{\rm{val}} X_i^{\rm{val}} "
    r"\hat \beta^{(\lambda)} } \right ) $")

plt.xscale("log")
plt.tick_params(width=5)
plt.legend()
plt.tight_layout()
plt.show(block=False)
