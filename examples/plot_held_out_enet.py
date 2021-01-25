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
import matplotlib.pyplot as plt
from sklearn import linear_model
from libsvmdata.datasets import fetch_libsvm
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error

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

# dataset = 'rcv1'
dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1_train')
    y -= y.mean()
    y /= np.linalg.norm(y)
else:
    X, y = make_regression(
        n_samples=20, n_features=100, noise=1, random_state=42)


n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
alpha_max = np.max(np.abs(X[idx_train, :].T @ y[idx_train])) / len(idx_train)

alpha_min = 1e-4 * alpha_max

n_grid = 10
alphas_L1 = np.geomspace(alpha_max, alpha_min, n_grid)
alphas_L2 = np.geomspace(alpha_max, alpha_min, n_grid)

results = np.zeros((n_grid, n_grid))
tol = 1e-5
max_iter = 10_000

estimator = linear_model.ElasticNet(
    fit_intercept=False, tol=tol, max_iter=max_iter, warm_start=True)

##############################################################################
# grid search with scikit-learn
# -----------------------------

print("Started grid search")
t_grid_search = - time.time()
for i in range(n_grid):
    print("lambda %i / %i" % (i * n_grid, n_grid * n_grid))
    for j in range(n_grid):
        estimator.alpha = (alphas_L1[i] + alphas_L2[j])
        estimator.l1_ratio = alphas_L1[i] / (alphas_L1[i] + alphas_L2[j])
        estimator.fit(X[idx_train, :], y[idx_train])
        results[i, j] = mean_squared_error(
            y[idx_val], estimator.predict(X[idx_val, :]))
t_grid_search += time.time()
print("Finished grid search")
print("Minimum outer criterion value with grid search %0.3e" % results.min())

##############################################################################
# Grad-search with sparse-ho
# --------------------------
estimator = linear_model.ElasticNet(
    fit_intercept=False, max_iter=max_iter, warm_start=True)
print("Started grad-search")
t_grad_search = - time.time()
monitor = Monitor()
n_outer = 25
log_alpha0 = np.array([np.log(alpha_max * 0.3), np.log(alpha_max / 10)])
model = ElasticNet(max_iter=max_iter, estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward(tol_jac=1e-3, n_iter_jac=100, max_iter=max_iter)
optimizer = GradientDescent(
    n_outer=n_outer, tol=tol, p_grad0=1.5, verbose=True)
grad_search(
    algo, criterion, model, optimizer, X, y, log_alpha0=log_alpha0,
    monitor=monitor)
t_grad_search += time.time()
monitor.log_alphas = np.array(monitor.log_alphas)

print("Time grid search %f" % t_grid_search)
print("Time grad-search %f" % t_grad_search)
print("Minimum grid search %0.3e" % results.min())
print("Minimum grad search %0.3e" % np.array(monitor.objs).min())

##############################################################################
# Plot results
# ------------

scaling_factor = results.max()
cmap = discrete_cmap(n_outer, 'Greens')
c = np.linspace(1, n_outer, n_outer)
X, Y = np.meshgrid(alphas_L1 / alpha_max, alphas_L2 / alpha_max)
fig, ax = plt.subplots(1, 1)
cp = ax.contourf(X, Y, results.T / scaling_factor)
ax.scatter(
    X, Y, s=10, c="orange", marker="o", label="$0$th order (grid search)",
    clip_on=False, cmap="viridis")
ax.scatter(
    np.exp(monitor.log_alphas[:, 0]) / alpha_max,
    np.exp(monitor.log_alphas[:, 1]) / alpha_max,
    s=50, cmap=cmap, c=c,
    marker="X", label="$1$st order", clip_on=False)
ax.set_xlim(X.min(), X.max())
ax.set_xlabel("L1 regularization")
ax.set_ylabel("L2 regularization")
ax.set_ylim(Y.min(), Y.max())
ax.set_title("Elastic net held out prediction loss on test set")
cb = fig.colorbar(cp)
cb.set_label("Held-out loss")
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.show(block=False)
