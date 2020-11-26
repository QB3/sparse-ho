"""
===========================
How to use custom metrics?
===========================

This example shows how to compute customize metrics using a callback function
as in scipy.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import time
import numpy as np
from numpy.linalg import norm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import linear_model
from sklearn.model_selection import train_test_split

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

# dataset = 'rcv1'
dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1_train')
else:
    X, y = make_regression(
        n_samples=1000, n_features=1000, noise=40, random_state=0)

# The dataset is split in 2: the data for training and validation: X
# Useen data X_test, asserting the quality of the model
X, X_test, y, y_test = train_test_split(X, y, test_size=0.333)

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train]))) / len(idx_train)
log_alpha0 = np.log(alpha_max / 10)

tol = 1e-7
max_iter = 1e5


estimator = linear_model.Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)

#############################################################################
# Call back definition
objs_test_grid = []


def callback_grid(val, grad, mask, dense, log_alpha):
    beta = np.zeros(len(mask))
    beta[mask] = dense
    # The custom quantity is added at each outer iteration:
    # here the loss on test data
    objs_test_grid.append(
        norm(X[np.ix_(idx_val, mask)] @ dense - y[idx_val]) ** 2 / len(idx_val))
    return np.array(objs_test_grid)


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
monitor_grid_sk = Monitor(callback=callback_grid)
grid_search(
    algo, criterion, model, X, y, None, None, monitor_grid_sk, log_alphas=log_alphas, tol=tol)
objs = np.array(monitor_grid_sk.objs)
t_sk = time.time() - t0

print('scikit-learn finished')


##############################################################################
# Grad-search with sparse-ho and callback
# ---------------------------------------

objs_test_grad = []


def callback_grad(val, grad, mask, dense, log_alpha):
    beta = np.zeros(len(mask))
    beta[mask] = dense
    # The custom quantity is added at each outer iteration:
    # here the loss on test data
    objs_test_grad.append(
        norm(X[np.ix_(idx_val, mask)] @ dense - y[idx_val]) ** 2 / len(idx_val))
    return np.array(objs_test_grad)


print('sparse-ho started')

t0 = time.time()
model = Lasso(estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward(criterion)
# use Monitor(callback)
monitor_grad = Monitor(callback=callback_grad)

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


current_palette = sns.color_palette("colorblind")

fig = plt.figure(figsize=(5, 3))
plt.semilogx(
    p_alphas, objs_test_grid, color=current_palette[0])
plt.semilogx(
    p_alphas, objs_test_grid, 'bo', label='0-order method (grid-search)',
    color=current_palette[1])
plt.semilogx(
    p_alphas_grad, objs_test_grad, 'bX', label='1-st order method',
    color=current_palette[2])
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel(
    r"$\|y^{\rm{test}} - X^{\rm{test}} \hat \beta^{(\lambda)} \|^2$")
plt.tick_params(width=5)
plt.legend()
plt.tight_layout()
plt.show(block=False)
