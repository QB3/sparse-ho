"""
============================
Lasso with held-out test set
============================

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
from mpl_toolkits.mplot3d import Axes3D

from libsvmdata.datasets import fetch_libsvm

from sklearn.datasets import make_regression
from sparse_ho import ImplicitForward
from sparse_ho.criterion import HeldOutMSE
from sparse_ho.models import ElasticNet
from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor

from sparse_ho.optimizers import LineSearch

Axes3D  # hack for matplotlib 3D support
# TODO improve example and remove this 3D graph

# dataset = "rcv1"
dataset = 'simu'
# use_small_part = False
use_small_part = True

##############################################################################
# Load some data

print("Started to load data")
dataset = 'rcv1'
# dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1_train')
else:
    X, y = make_regression(n_samples=1000, n_features=1000, noise=40)

print("Finished loading data")

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train]))) / len(idx_train)
log_alpha_max = np.log(alpha_max)

alpha_min = 1e-4 * alpha_max

n_grid = 10
alphas_1 = np.geomspace(0.6 * alpha_max, alpha_min, n_grid)
log_alphas_1 = np.log(alphas_1)
alphas_2 = np.geomspace(0.6 * alpha_max, alpha_min, n_grid)
log_alphas_2 = np.log(alphas_2)

results = np.zeros((n_grid, n_grid))
tol = 1e-4
max_iter = 50000

estimator = linear_model.ElasticNet(
    fit_intercept=False, tol=tol, max_iter=max_iter, warm_start=True)

##############################################################################
# Grid-search with scikit-learn
# -----------------------------

print("Started grid-search")
t_grid_search = - time.time()
for i in range(n_grid):
    print("lambda %i / %i" % (i, n_grid))
    for j in range(n_grid):
        print("lambda %i / %i" % (j, n_grid))
        estimator.alpha = (alphas_1[i] + alphas_2[j])
        estimator.l1_ratio = alphas_1[i] / (alphas_1[i] + alphas_2[j])
        estimator.fit(X[idx_train, :], y[idx_train])
        results[i, j] = np.mean((y[idx_val] - X[idx_val, :] @ estimator.coef_) ** 2)
t_grid_search += time.time()
print("Finished grid-search")


##############################################################################
# Grad-search with sparse-ho
# --------------------------
estimator = linear_model.ElasticNet(
    fit_intercept=False, max_iter=max_iter, warm_start=True)
print("Started grad-search")
t_grad_search = - time.time()
monitor = Monitor()
n_outer = 10
log_alpha0 = np.array([np.log(alpha_max * 0.3), np.log(alpha_max / 10)])
model = ElasticNet(max_iter=max_iter, estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward(tol_jac=1e-7, n_iter_jac=1000, max_iter=max_iter)
optimizer = LineSearch(n_outer=n_outer, tol=tol, verbose=True)
grad_search(
    algo, criterion, model, optimizer, X, y, log_alpha0=log_alpha0,
    monitor=monitor)
t_grad_search += time.time()
alphas_grad = np.exp(np.array(monitor.log_alphas))
alphas_grad /= alpha_max


print("Time grid-search %f" % t_grid_search)
print("Time grad-search %f" % t_grad_search)
print("Minimum grid search %0.3e" % results.min())
print("Minimum grad search %0.3e" % np.array(monitor.objs).min())

##############################################################################
# Plot results
# ------------

idx = np.where(results == results.min())

a, b = np.meshgrid(alphas_1 / alpha_max, alphas_2 / alpha_max)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(
    np.log(a), np.log(b), results, rstride=1, cstride=1,
    cmap='viridis', edgecolor='none', alpha=0.5)
ax.scatter3D(
    np.log(a), np.log(b), results,
    monitor.objs, c="black", s=20, marker="o")
ax.scatter3D(
    np.log(alphas_grad[:, 0]), np.log(alphas_grad[:, 1]),
    monitor.objs, c="red", s=200, marker="X")
ax.scatter3D(
    np.log(alphas_2[idx[1]] / alpha_max),
    np.log(alphas_1[idx[0]] / alpha_max),
    [results.min()], c="black", s=200, marker="X")
ax.set_xlabel("lambda1")
ax.set_ylabel("lambda2")
ax.set_label("Loss on validation set")
fig.show()
