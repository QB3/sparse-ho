"""
=============================
Expe Lasso
=============================

File to play with expes for the Lasso
"""

import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import norm

from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
import celer
from libsvmdata import fetch_libsvm

from sparse_ho import ImplicitForward
from sparse_ho import grad_search, hyperopt_wrapper
from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import GradientDescent
from sparse_ho.utils import Monitor
from sparse_ho.utils_plot import configure_plt
from sparse_ho.grid_search import grid_search

configure_plt()

dataset = 'rcv1_train'
# dataset = 'simu'

if dataset != 'simu':
    X, y = fetch_libsvm(dataset)
    y -= y.mean()
    y /= norm(y)
else:
    X, y = make_regression(
        n_samples=500, n_features=1000, noise=40,
        random_state=42)

n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
alpha_min = alpha_max / 10_000


tol = 1e-8

estimator = celer.Lasso(
    fit_intercept=False, max_iter=100, warm_start=True, tol=tol)


dict_monitor = {}

all_algo_name = ['implicit_forward', 'grid_search']

for algo_name in all_algo_name:
    model = Lasso(estimator=estimator)
    sub_criterion = HeldOutMSE(None, None)
    alpha0 = alpha_max / 10
    monitor = Monitor()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    criterion = CrossVal(sub_criterion, cv=kf)
    algo = ImplicitForward(tol_jac=1e-3)
    optimizer = GradientDescent(
        n_outer=30, p_grad_norm=1., verbose=True, tol=tol)
    if algo_name == 'implicit_forward':
        grad_search(
            algo, criterion, model, optimizer, X, y, alpha0,
            monitor)
    elif algo_name == 'grid_search':
        grid_search(
            algo, criterion, model, X, y, alpha_min, alpha_max,
            monitor, max_evals=20, tol=tol)
    elif algo_name == 'random_search':
        hyperopt_wrapper(
            algo, criterion, model, X, y, alpha_min, alpha_max,
            monitor, max_evals=20, tol=tol, method='random', size_space=1)
    dict_monitor[algo_name] = monitor


min_objs = np.infty
for monitor in dict_monitor.values():
    monitor.objs = np.array(monitor.objs)
    min_objs = min(min_objs, monitor.objs.min())

scaling_factor = (y @ y) / len(y)
plt.figure()
for monitor in dict_monitor.values():
    plt.plot(monitor.times, monitor.objs / scaling_factor)
plt.xlabel('Time (s)')
plt.ylabel('Objective')
plt.show(block=False)

plt.figure()
for monitor in dict_monitor.values():
    plt.semilogy(monitor.times, (monitor.objs - min_objs) / scaling_factor)
plt.xlabel('Time (s)')
plt.ylabel('Objective - optimum')
plt.show(block=False)

plt.figure()
monitor_grid = dict_monitor['grid_search']
plt.semilogx(
    monitor_grid.alphas / alpha_max,
    monitor_grid.objs / scaling_factor)
for monitor in dict_monitor.values():
    plt.scatter(
        monitor.alphas / alpha_max, monitor.objs / scaling_factor, marker='X')
plt.xlabel('alpha')
plt.ylabel('Objective')
plt.show(block=False)
