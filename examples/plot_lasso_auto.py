"""
============================
Lasso with held-out test set
============================

This example shows how to perform hyperparameter optimization
for a Lasso using a held-out validation set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#          Mathurin Massias
#
# License: BSD (3-clause)

import celer
import numpy as np
from celer.datasets import make_correlated_data
from libsvmdata.datasets import fetch_libsvm

from sparse_ho.criterion import HeldOutMSE
from sparse_ho import ImplicitForward
from sparse_ho.optimizers import GradientDescent
from sparse_ho.wrappers.wrapper_lasso import LassoAuto

print(__doc__)

# dataset = 'rcv1'
dataset = 'simu'

if dataset == 'rcv1':
    X, y = fetch_libsvm('rcv1_train')
else:
    X, y, _ = make_correlated_data(
        n_samples=100, n_features=200, random_state=0)

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train])))
alpha_max /= len(idx_train)
alpha0 = alpha_max / 5

tol = 1e-7

estimator = celer.Lasso(fit_intercept=False, warm_start=True)


##############################################################################
# Grad-search with sparse-ho
# --------------------------

print('sparse-ho started')

criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward()
optimizer = GradientDescent(n_outer=30, tol=tol, verbose=True)
lasso_auto = LassoAuto(estimator, criterion, alpha0=alpha0)
lasso_auto.fit(X, y, algo, optimizer=optimizer)

print('sparse-ho finished')
