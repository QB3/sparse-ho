# Experiment for Lasso CV
# License: BSD (3-clause)

import numpy as np
import celer

from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from celer import LassoCV
from sklearn.model_selection import KFold

from sparse_ho import ImplicitForward
from sparse_ho import grad_search, hyperopt_wrapper
from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import GradientDescent
from sparse_ho.utils import Monitor

print(__doc__)

dataset = 'real-sim'
# dataset = 'rcv1_train'

if dataset != 'simu':
    X, y = fetch_libsvm(dataset)
    y -= np.mean(y)
    y /= np.std(y)
else:
    X, y = make_regression(
        n_samples=500, n_features=1000, noise=40, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples


tol = 1e-3
max_iter = 100_000

algorithms = ['bayesian']
# ['grid_search100', 'grid_search10', 'grad_search', 'random', 'bayesian']

p_alpha_min = 1 / 10_000
print("Starting path computation...")
for algorithm in algorithms:
    estimator = celer.Lasso(
        fit_intercept=False, max_iter=1000, warm_start=True, tol=tol,
        verbose=True)

    print('%s started' % algorithm)

    model = Lasso(estimator=estimator)
    criterion = HeldOutMSE(None, None)
    alpha0 = alpha_max / 10
    monitor = Monitor()
    cross_val_criterion = CrossVal(criterion, cv=kf)
    algo = ImplicitForward()
    optimizer = GradientDescent(
        n_outer=10, tol=tol, verbose=True, p_grad_norm=1)
    # optimizer = LineSearch(n_outer=10, tol=tol, verbose=True)
    if algorithm == 'grad_search':
        grad_search(
            algo, cross_val_criterion, model, optimizer, X, y, alpha0,
            monitor)
        objs = np.array(monitor.objs)
        log_alphas = np.log(np.array(monitor.alphas) / alpha_max)

    elif algorithm.startswith('grid_search'):
        if algorithm == 'grid_search10':
            n_alphas = 10
        else:
            n_alphas = 100
        p_alphas = np.geomspace(1, p_alpha_min, n_alphas)
        alphas = alpha_max * p_alphas
        reg = LassoCV(
            cv=kf, verbose=True, tol=tol, fit_intercept=False,
            alphas=alphas, max_iter=max_iter).fit(X, y)
        reg.score(X, y)
        objs = reg.mse_path_.mean(axis=1)
        log_alphas = np.log(alphas / alpha_max)
    else:
        hyperopt_wrapper(
            algo, cross_val_criterion, model, X, y,
            alpha_max * p_alpha_min,
            alpha_max, monitor, max_evals=10,
            method=algorithm, size_space=1, tol=tol, random_state=4)
        objs = np.array(monitor.objs)
        log_alphas = np.log(np.array(monitor.alphas) / alpha_max)
    np.save("results/log_alphas_%s" % algorithm, log_alphas)
    np.save("results/objs_%s" % algorithm, objs)
    print('%s finished' % algorithm)
