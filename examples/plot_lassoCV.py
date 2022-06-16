"""
=============================
Lasso with Cross-validation
=============================

This example shows how to perform hyperparameter optimization
for a Lasso using a full cross-validation score.
"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#          Mathurin Massias

# License: BSD (3-clause)

import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from celer import LassoCV
from sklearn.model_selection import KFold

from sparse_ho import ImplicitForward, grad_search
from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import GradientDescent
from sparse_ho.utils import Monitor
from sparse_ho.utils_plot import discrete_cmap

print(__doc__)

# dataset = 'rcv1'
dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1.binary')
else:
    X, y = make_regression(
        n_samples=500, n_features=1000, noise=40,
        random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

print("Starting path computation...")
n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples

n_alphas = 10
alphas = np.geomspace(alpha_max, alpha_max / 1_000, n_alphas)

tol = 1e-8

#############################################################################
# Cross-validation with scikit-learn
# ----------------------------------
print('scikit started')

t0 = time.time()
reg = LassoCV(
    cv=kf, verbose=True, tol=tol, fit_intercept=False,
    alphas=alphas, max_iter=100_000).fit(X, y)
reg.score(X, y)
t_sk = time.time() - t0

print('scikit finished')

##############################################################################
# Now do the hyperparameter optimization with implicit differentiation
# --------------------------------------------------------------------

estimator = sklearn.linear_model.Lasso(fit_intercept=False,
                                       warm_start=True, max_iter=100_000)

print('sparse-ho started')

t0 = time.time()
model = Lasso(estimator)
criterion = HeldOutMSE(None, None)
alpha0 = 0.9 * alpha_max
monitor_grad = Monitor()
cross_val_criterion = CrossVal(criterion, cv=kf)
algo = ImplicitForward()
optimizer = GradientDescent(n_outer=10, tol=tol)
grad_search(
    algo, cross_val_criterion, model, optimizer, X, y, alpha0,
    monitor_grad)

t_grad_search = time.time() - t0

print('sparse-ho finished')

##############################################################################
# Plot results
# ------------
objs = reg.mse_path_.mean(axis=1)

p_alphas_grad = np.array(monitor_grad.alphas) / alpha_max
objs_grad = np.array(monitor_grad.objs)


print(f"Time for grid search: {t_sk:.2f} s")
print(f"Time for grad search (sparse-ho): {t_grad_search:.2f} s")

print(f'Minimum outer criterion value with grid search: {objs.min():.5f}')
print(f'Minimum outer criterion value with grad search: {objs_grad.min():.5f}')

current_palette = sns.color_palette("colorblind")
cmap = discrete_cmap(len(objs_grad), 'Greens')

fig, ax = plt.subplots(figsize=(5, 3))
ax.plot(alphas / alphas[0], objs, color=current_palette[0])
ax.plot(
    alphas / alphas[0], objs,
    'bo', label='0-th order method (grid search)',
    color=current_palette[1])
ax.scatter(
    p_alphas_grad, objs_grad,
    label='1-st order method',  marker='X',
    color=cmap(np.linspace(0, 1, len(objs_grad))), s=40, zorder=40)
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel("Cross-validation loss")
ax.set_xscale("log")
plt.tick_params(width=5)
plt.legend()
plt.tight_layout()
plt.show(block=False)
