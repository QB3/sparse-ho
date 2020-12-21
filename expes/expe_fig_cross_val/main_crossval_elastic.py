# Experiment for Lasso CV
# License: BSD (3-clause)

import time
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import celer
import sklearn

from libsvmdata import fetch_libsvm
from sklearn.datasets import make_regression
from celer import LassoCV
# from sklearn.linear_model import LassoCV
from sklearn.model_selection import KFold

from sparse_ho import ImplicitForward
from sparse_ho import grad_search, hyperopt_wrapper
from sparse_ho.models import ElasticNet
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import LineSearch, GradientDescent
from sparse_ho.utils import Monitor
from sparse_ho.grid_search import grid_search


print(__doc__)

dataset = 'real-sim'
# dataset = 'rcv1_train'
# dataset = 'simu'

if dataset != 'simu':
    X, y = fetch_libsvm(dataset)
    y -= np.mean(y)
else:
    X, y = make_regression(
        n_samples=500, n_features=1000, noise=40, random_state=42)

kf = KFold(n_splits=5, shuffle=True, random_state=42)

n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
log_alpha_max = np.log(alpha_max)
p_alpha_min = alpha_max / 10_000
log_alpha_min = np.log(p_alpha_min * alpha_max)

tol = 1e-8
max_iter = 1e5

# algorithms = ['grid_search100']
# algorithms = ['grad_search']
# algorithms = [
    # 'grid_search10', 'grid_search100', 'grad_search', 'random', 'bayesian']
# algorithms = [
#     'grid_search10', 'grad_search', 'random', 'bayesian', 'grid_search100']
algorithms = ['grad_search_ls']

max_evals = 25
print("Starting path computation...")
for algorithm in algorithms:
    estimator = sklearn.linear_model.ElasticNet(
        fit_intercept=False, max_iter=3_000, warm_start=True, tol=tol)

    print('%s started' % algorithm)

    model = ElasticNet(estimator=estimator)
    criterion = HeldOutMSE(None, None)
    log_alpha0 = np.array([np.log(alpha_max / 10), np.log(alpha_max / 10)])
    monitor = Monitor()
    cross_val_criterion = CrossVal(criterion, cv=kf)
    algo = ImplicitForward()
    # optimizer = LineSearch(n_outer=10, tol=tol, verbose=True)
    if algorithm.startswith('grad_search'):
        if algorithm == 'grad_search':
            optimizer = GradientDescent(
                n_outer=max_evals, tol=tol, verbose=True, p_grad0=1)
        else:
            optimizer = LineSearch(n_outer=25, verbose=True, tol=tol)
        grad_search(
            algo, cross_val_criterion, model, optimizer, X, y, log_alpha0,
            monitor)

    elif algorithm.startswith('grid_search'):
        if algorithm == 'grid_search10':
            n_alphas = 5
        else:
            n_alphas = 100
        p_alphas = np.geomspace(1, p_alpha_min, n_alphas)
        alphas = alpha_max * p_alphas
        log_alphas = np.log(alphas)
        grid_alphas = [i for i in itertools.product(log_alphas, log_alphas)]

        grid_search(
            algo, cross_val_criterion, model, X, y, None, None, monitor,
            log_alphas=grid_alphas)
        # reg = LassoCV(
        #     cv=kf, verbose=True, tol=tol, fit_intercept=False,
        #     alphas=alphas, max_iter=max_iter).fit(X, y)
        # reg.score(X, y)
        # objs = reg.mse_path_.mean(axis=1)
        # log_alphas = np.log(alphas)
    else:
        hyperopt_wrapper(
            algo, cross_val_criterion, model, X, y, log_alpha_min,
            log_alpha_max, monitor, max_evals=max_evals,
            method=algorithm, size_space=2)

    objs = np.array(monitor.objs)
    log_alphas = np.array(monitor.log_alphas)
    log_alphas -= np.log(alpha_max)
    np.save("results/%s_log_alphas_%s_enet" % (dataset, algorithm), log_alphas)
    np.save("results/%s_objs_%s_enet" % (dataset, algorithm), objs)
    print('%s finished' % algorithm)
