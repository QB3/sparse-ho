"""
=====================================
Weighted Lasso with held-out test set
=====================================

This example shows how to perform hyperparameter optimization
for a weighted Lasso using a held-out validation set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import time

import numpy as np

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split

from celer import Lasso

from sparse_ho.models import wLasso
from sparse_ho.criterion import CV
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search
from sparse_ho.datasets import get_leukemia

print(__doc__)

dataset = 'leukemia'
# dataset = 'simu'

if dataset == 'leukemia':
    X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()
else:
    X, y = make_regression(n_samples=100, n_features=100, noise=1)
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
# Grid-search
# -----------

# the solver of sklearn is indeed very long on the considered problems!
# estimator = linear_model.Lasso(
#     fit_intercept=False, max_iter=1000, warm_start=True)

# celer is much more faster !
# https://github.com/mathurinm/celer

estimator = Lasso(
    fit_intercept=False, max_iter=1000, warm_start=True)


##############################################################################
# Grad-search
# -----------
print('sparse-ho started')

alpha0 = np.log(alpha_max / 10) * np.ones(n_features)

t0 = time.time()
model = wLasso(X_train, y_train, estimator=estimator)

# here CV means held out
# the "real" crossval (with folds etc) is very slow (for the moment) for some
# unknown reasons

criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = ImplicitForward(criterion)
monitor_grad = Monitor()
grad_search(
    algo, alpha0, monitor_grad, n_outer=10, tol=tol)

t_grad_search = time.time() - t0

print("Time gradient serach:  %f" % t_grad_search)
