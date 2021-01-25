"""
==================================
Elastic net with held-out test set
==================================

This example shows how to perform hyperparameter optimization
for an elastic-net using a held-out validation set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import time
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

from libsvmdata.datasets import fetch_libsvm

from sklearn.datasets import make_regression
from sparse_ho import ImplicitForward
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.models import ElasticNet
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.utils_plot import discrete_cmap
from sparse_ho.optimizers import GradientDescent


# dataset = "rcv1"
dataset = 'simu'

##############################################################################
# Load some data

print("Started to load data")
# dataset = 'rcv1'
dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1_train')
    # X = X[:1000, :]
    y -= y.mean()
    y /= np.linalg.norm(y)
else:
    X, y = make_regression(
        n_samples=20, n_features=100, noise=1, random_state=42)

print("Finished loading data")

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train])))
alpha_max /= len(idx_train)

alpha_min = 1e-4 * alpha_max

n_grid = 10
alphas_1 = np.geomspace(alpha_max, alpha_min, n_grid)
alphas_2 = np.geomspace(alpha_max, alpha_min, n_grid)

results = np.zeros((n_grid, n_grid))
tol = 1e-5
max_iter = 10_000

estimator = linear_model.ElasticNet(
    fit_intercept=False, tol=tol, max_iter=max_iter, warm_start=True)

##############################################################################
# Grid-search with scikit-learn
# -----------------------------

print("Started grid-search")
t_grid_search = - time.time()
for i in range(n_grid):
    print("lambda %i / %i" % (i, n_grid * n_grid))
    for j in range(n_grid):
        estimator.alpha = (alphas_1[i] + alphas_2[j])
        estimator.l1_ratio = alphas_1[i] / (alphas_1[i] + alphas_2[j])
        estimator.fit(X[idx_train, :], y[idx_train])
        results[i, j] = np.mean(
            (y[idx_val] - X[idx_val, :] @ estimator.coef_) ** 2)
t_grid_search += time.time()
print("Finished grid-search")
print("Minimum grid search %0.3e" % results.min())

##############################################################################
# Grad-search with sparse-ho
# --------------------------
estimator = linear_model.ElasticNet(
    fit_intercept=False, max_iter=max_iter, warm_start=True)
print("Started grad-search")
t_grad_search = - time.time()
monitor = Monitor()
n_outer = 25
alpha0 = np.array([alpha_max * 0.3, alpha_max / 10])
model = ElasticNet(max_iter=max_iter, estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward(tol_jac=1e-3, n_iter_jac=100, max_iter=max_iter)
optimizer = GradientDescent(
    n_outer=n_outer, tol=tol, p_grad0=1.5, verbose=True)
grad_search(
    algo, criterion, model, optimizer, X, y, alpha0=alpha0,
    monitor=monitor)
t_grad_search += time.time()
monitor.alphas = np.array(monitor.alphas)

print("Time grid-search %f" % t_grid_search)
print("Time grad-search %f" % t_grad_search)
print("Minimum grid search %0.3e" % results.min())
print("Minimum grad search %0.3e" % np.array(monitor.objs).min())

##############################################################################
# Plot results
# ------------

scaling_factor = results.max()
cmap = discrete_cmap(n_outer, 'Greens')
c = np.linspace(1, n_outer, n_outer)
X, Y = np.meshgrid(alphas_1 / alpha_max, alphas_2 / alpha_max)
fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, results.T / scaling_factor)
ax.scatter(
    X, Y, s=10, c="orange", marker="o", label="$0$ order (grid search)",
    clip_on=False, cmap="viridis")
ax.scatter(
    monitor.alphas[:, 0]) / alpha_max,
    monitor.alphas[:, 1]) / alpha_max,
    s=50, cmap=cmap, c=c,
    marker="X", label="$1$st order", clip_on=False)
ax.set_xlim(X.min(), X.max())
ax.set_ylim(Y.min(), Y.max())
cb=fig.colorbar(cp)
cb.set_label("Held-out loss")
plt.xscale('log')
plt.yscale('log')
plt.show(block=False)
