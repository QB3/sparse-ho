"""
Method comparison on Lasso
==========================

The aim of this example is to demonstrate on a simple
dateset how methods compare.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor, WarmStart
from sparse_ho.grid_search import grid_search
# from sparse_ho.bayesian import hyperopt_lasso

from sparse_ho.datasets.real import get_20newsgroup
from expes.utils import configure_plt

print(__doc__)

# from sparse_ho.datasets.real import get_leukemia
# X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()

# from sparse_ho.datasets.real import get_finance
# X_train, X_val, X_test, y_train, y_val, y_test = get_finance()

X_train, X_val, X_test, y_train, y_val, y_test = get_20newsgroup()

n_samples, n_features = X_train.shape

alpha_max = (np.abs(X_train.T @ y_train)).max() / n_samples
maxit = 1000
n_alpha = 100
p_alphas = np.geomspace(1, 0.0001, n_alpha)
log_alphas = np.log(alpha_max * p_alphas)


alpha0 = 0.6 * alpha_max
# alpha0 = 0.1 * alpha_max
log_alpha0_mcp = np.array([np.log(alpha0), np.log(100)])
log_alpha0 = np.log(alpha0)
log_alpha_min = np.log(alpha_max / 100)
log_alpha_max = np.log(alpha_max)

tol = 1e-5

dict_times = {}
dict_pobj = {}

# methods = ["forward", "implicit_forward"]
# methods = ["forward", "implicit_forward", "grid_search"]
methods = ["forward", "implicit_forward", "implicit", "grid_search"]
maxits = np.floor(np.geomspace(5, 100, 5)).astype(int)


model = "lasso"

dict_n_outers = {}
dict_n_outers["20newsgroups", "implicit_forward"] = 50
dict_n_outers["20newsgroups", "forward"] = 30
dict_n_outers["20newsgroups", "implicit"] = 30
dict_n_outers["20newsgroups", "bayesian"] = 75
dict_n_outers["20newsgroups", "random"] = 35

dataset_name = "20newsgroups"

for method in methods:
    monitor = Monitor()
    warm_start = WarmStart()
    if method == "grid_search":
        grid_searchCV(
            X_train, y_train, log_alphas, X_val, y_val, X_test, y_test,
            tol, monitor)
    else:
        n_outer = dict_n_outers[dataset_name, method]
        grad_search(
            X_train, y_train, log_alpha0, X_val, y_val, X_test, y_test, tol,
            monitor, method=method, maxit=1000, n_outer=n_outer,
            warm_start=warm_start, model="lasso", t_max=20)
    pobj = np.array(
        [np.min(monitor.objs[:k]) for k in np.arange(
            len(monitor.objs)) + 1])
    dict_pobj[method] = pobj
    dict_times[method] = np.array(monitor.times)


pobj_star = np.infty
for method in methods:
    pobj_star = np.minimum(pobj_star, np.min(dict_pobj[method]))

dict_legend = {}
dict_legend["implicit_forward"] = "implicit forward"
dict_legend["forward"] = "forward"
dict_legend["implicit"] = "implicit"
dict_legend["grid_search"] = "grid search"


configure_plt()
current_palette = sns.color_palette("colorblind")
plt.figure()
for method in methods:
    plt.semilogy(
        dict_times[method], dict_pobj[method] - pobj_star,
        label=dict_legend[method], marker="X")
plt.ylabel("Objective minus optimum")
plt.xlabel("Time")
plt.legend()
plt.show(block=False)
