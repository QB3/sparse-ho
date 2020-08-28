"""
This is the file to reproduce the experiments of Figure 2:
'Computation time for the HO of the Lasso on real data.'
It is recommended to run this script on a cluster with several CPUs.
"""

import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
from itertools import product
import pandas as pd

from sparse_ho.datasets.real import get_data

from sparse_ho.models import Lasso
from sparse_ho.criterion import CV
from sparse_ho.utils import Monitor

from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.implicit import Implicit
from sparse_ho.grid_search import grid_search
# from sparse_ho.bayesian import hyperopt_lasso

from sparse_ho.ho import grad_search

#######################################################################
n_jobs = 1
# n_jobs = len(dataset_names) * len(methods) * len(tolerance_decreases)
#######################################################################



#######################################################################
dataset_names = ["rcv1"]
# dataset_names = ["20newsgroups"]
# dataset_names = ["finance"]
# uncomment the following line to launch the experiments on other
# datasets:
# dataset_names = ["rcv1", "20newsgroups", "finance"]
methods = [
    "implicit_forward",
    # "implicit",
    "forward", 'grid_search',
    "random"]  # "bayesian",
tolerance_decreases = ["constant"]
tols = [1e-7]
n_outers = [75]


dict_n_outers = {}
dict_n_outers["20newsgroups", "implicit_forward"] = 50
dict_n_outers["20newsgroups", "forward"] = 60
dict_n_outers["20newsgroups", "implicit"] = 6
dict_n_outers["20newsgroups", "bayesian"] = 75
dict_n_outers["20newsgroups", "random"] = 35

dict_n_outers["finance", "implicit_forward"] = 125
dict_n_outers["finance", "forward"] = 75
dict_n_outers["finance", "implicit"] = 6
dict_n_outers["finance", "bayesian"] = 75
dict_n_outers["finance", "random"] = 50

dict_algo = {}
dict_algo["implicit_forward"] = ImplicitForward
dict_algo["implicit"] = Implicit
dict_algo["forward"] = Forward


def parallel_function(
        dataset_name, method, tol=1e-5, n_outer=50,
        tolerance_decrease='constant'):

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(dataset_name)
    n_samples, _ = X_train.shape
    # compute alpha_max
    alpha_max = np.abs(X_train.T @ y_train).max() / n_samples
    log_alpha_max = np.log(alpha_max)
    log_alpha_min = np.log(alpha_max/1000)
    log_alpha0 = np.log(0.1 * alpha_max)

    model = Lasso(X_train, y_train)
    criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)

    try:
        n_outer = dict_n_outers[dataset_name, method]
    except Exception:
        n_outer = 20

    if dataset_name == "rcv1":
        size_loop = 2
    else:
        size_loop = 1

    for _ in range(size_loop):
        monitor = Monitor()
        if method == 'grid_search':
            algo = Forward(criterion)
            log_alphas = np.log(np.geomspace(
                alpha_max, alpha_max/1000, num=100))
            grid_search(
                algo, None, None, monitor, log_alphas=log_alphas,
                tol=tol)
        elif method == 'random':
            algo = Forward(criterion)
            grid_search(
                algo, log_alpha_max, log_alpha_min, monitor, tol=tol, max_evals=n_outer)
        elif method in ("bayesian", "random"):
            # TODO
            1 / 0
            # monitor = hyperopt_lasso(
            #     X_train, y_train, log_alpha0, X_val, y_val, X_test,
            #     y_test, tol, max_evals=n_outer, method=method)
        else:
            # do line search to find the optimal lambda
            # import ipdb; ipdb.set_trace()
            algo = dict_algo[method](criterion)
            grad_search(
                algo, log_alpha0, monitor, n_outer=n_outer, tol=tol)

        monitor.times = np.array(monitor.times)
        monitor.objs = np.array(monitor.objs)
        monitor.objs_test = np.array(monitor.objs_test)
        monitor.log_alphas = np.array(monitor.log_alphas)
    return (dataset_name, method, tol, n_outer, tolerance_decrease,
            monitor.times, monitor.objs, monitor.objs_test,
            monitor.log_alphas, norm(y_val), norm(y_test))


print("enter sequential")
backend = 'loky'
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(parallel_function)(
        dataset_name, method, n_outer=n_outer,
        tolerance_decrease=tolerance_decrease)
    for dataset_name, method, n_outer,
    tolerance_decrease in product(
        dataset_names, methods, n_outers, tolerance_decreases))
print('OK finished parallel')

df = pd.DataFrame(results)
df.columns = [
    'dataset', 'method', 'tol', 'n_outer', 'tolerance_decrease',
    'times', 'objs', 'objs_test', 'log_alphas', 'norm y_val',
    'norm y_test']

for dataset_name in dataset_names:
    df[df['dataset'] == dataset_name].to_pickle(
        "%s.pkl" % dataset_name)
