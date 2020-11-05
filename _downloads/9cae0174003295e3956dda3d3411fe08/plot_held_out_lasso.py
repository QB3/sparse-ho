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
from sparse_ho.criterion import CV
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search
from sparse_ho.grid_search import grid_search
from sklearn.datasets import make_regression

from sklearn.model_selection import train_test_split


from sparse_ho.datasets import get_data

print(__doc__)

dataset = 'rcv1'
# dataset = 'simu'

if dataset == 'rcv1':
    X_train, X_val, X_test, y_train, y_val, y_test = get_data('rcv1_train')
else:
    X, y = make_regression(n_samples=1000, n_features=1000, noise=40)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.5)

n_samples, n_features = X_train.shape

print("Starting path computation...")
n_samples = len(y_train)
alpha_max = np.max(np.abs(X_train.T.dot(y_train))) / X_train.shape[0]
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
model = Lasso(X_train, y_train, estimator=estimator)
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = Forward(criterion)
monitor_grid_sk = Monitor()
grid_search(
    algo, None, None, monitor_grid_sk, log_alphas=log_alphas,
    tol=tol)
objs = np.array(monitor_grid_sk.objs)
t_sk = time.time() - t0

print('scikit-learn finished')


##############################################################################
# Grad-search with sparse-ho
# --------------------------

print('sparse-ho started')

t0 = time.time()
model = Lasso(X_train, y_train, estimator=estimator)
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = ImplicitForward(criterion)
monitor_grad = Monitor()
grad_search(
    algo, np.log(alpha_max / 10), monitor_grad, n_outer=10, tol=tol)

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
