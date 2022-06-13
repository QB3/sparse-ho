"""
=============================
Expe enet
=============================

File to play with expes for the enet
"""

from itertools import product
import numpy as np
import matplotlib.pyplot as plt

import celer
from sklearn.datasets import make_regression
from sklearn.model_selection import KFold
from libsvmdata import fetch_libsvm

from sparse_ho import ImplicitForward
from sparse_ho import grad_search, hyperopt_wrapper
from sparse_ho.models import ElasticNet
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.optimizers import GradientDescent
from sparse_ho.utils import Monitor
from sparse_ho.utils_plot import configure_plt
from sparse_ho.grid_search import grid_search
from sparse_ho.utils_plot import discrete_color

configure_plt()

# dataset = 'real-sim'
dataset = 'rcv1_train'
# dataset = 'simu'

if dataset != 'simu':
    X, y = fetch_libsvm(dataset)
    y -= y.mean()
else:
    X, y = make_regression(
        n_samples=500, n_features=1000, noise=40,
        random_state=42)

n_samples = len(y)
alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
alpha_min = alpha_max / 100_000

num1D = 5
alpha1D = np.geomspace(alpha_max, alpha_min, num=num1D)
alphas = [np.array(i) for i in product(alpha1D, alpha1D)]

tol = 1e-3

estimator = celer.ElasticNet(
    fit_intercept=False, max_iter=1000, warm_start=True, tol=tol)


dict_monitor = {}

all_algo_name = ['grid_search']
# , 'implicit_forward', "implicit_forward_approx", 'bayesian']
# , 'random_search']
# all_algo_name = ['random_search']

for algo_name in all_algo_name:
    model = ElasticNet(estimator=estimator)
    sub_criterion = HeldOutMSE(None, None)
    alpha0 = np.array([alpha_max / 10, alpha_max / 10])
    monitor = Monitor()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    criterion = CrossVal(sub_criterion, cv=kf)
    algo = ImplicitForward(tol_jac=1e-3)
    # optimizer = LineSearch(n_outer=10, tol=tol)
    if algo_name.startswith('implicit_forward'):
        if algo_name == "implicit_forward_approx":
            optimizer = GradientDescent(
                n_outer=30, p_grad_norm=1., verbose=True, tol=tol,
                tol_decrease="geom")
        else:
            optimizer = GradientDescent(
                n_outer=30, p_grad_norm=1., verbose=True, tol=tol)
        grad_search(
            algo, criterion, model, optimizer, X, y, alpha0,
            monitor)
    elif algo_name == 'grid_search':
        grid_search(
            algo, criterion, model, X, y, None, None,
            monitor, max_evals=20, tol=tol, alphas=alphas)
    elif algo_name == 'random_search' or algo_name == 'bayesian':
        hyperopt_wrapper(
            algo, criterion, model, X, y, alpha_min, alpha_max,
            monitor, max_evals=20, tol=tol, method=algo_name, size_space=2)
    else:
        1 / 0
    dict_monitor[algo_name] = monitor


min_objs = np.infty
for monitor in dict_monitor.values():
    monitor.objs = np.array(monitor.objs)
    min_objs = min(min_objs, monitor.objs.min())

scaling_factor = (y @ y) / len(y)
plt.figure()
for monitor in dict_monitor.values():
    obj = monitor.objs
    obj = [np.min(obj[:k]) for k in np.arange(len(obj)) + 1]
    plt.plot(monitor.times, obj / scaling_factor)
plt.xlabel('Time (s)')
plt.ylabel('Objective')
plt.show(block=False)


dict_colors = {
    'implicit_forward': 'Greens',
    'implicit_forward_approx':  'OrRd',
    'grid_search': 'Reds',
    'bayesian': 'Blues'
}

##############################################################


results = dict_monitor["grid_search"].objs.reshape(len(alpha1D), -1)
scaling_factor = results.max()
levels = np.geomspace(min_objs / scaling_factor, 1, num=20)

X, Y = np.meshgrid(alpha1D / alpha_max, alpha1D / alpha_max)
fig, ax = plt.subplots(1, 1)
cp = ax.contourf(
    X, Y, results.T / scaling_factor, levels=levels, cmap="viridis")
ax.scatter(
    X, Y, s=10, c="orange", marker="o", label="$0$th order (grid search)",
    clip_on=False)


for method in dict_monitor.keys():
    if method != 'grid_search':
        monitor = dict_monitor[method]
        monitor.alphas = np.array(monitor.alphas)
        n_outer = len(monitor.objs)
        color = discrete_color(n_outer, dict_colors[method])
        ax.scatter(
            monitor.alphas[:, 0] / alpha_max,
            monitor.alphas[:, 1] / alpha_max,
            s=50, color=color,
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
