"""
This is the file to reproduce the experiments of Figure 5 in Appendix:
'Computation time for the HO of the MCP on real data'.
It is recommended to run this script on a cluster with several CPUs.
"""

import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
import pandas
import scipy

from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor, WarmStart
from sparse_ho.grid_search import grid_searchMCP
from sparse_ho.bayesian import hyperopt_lasso
from itertools import product

from sparse_ho.datasets.real import get_data


dataset_names = ["rcv1"]
# uncomment the following line to launch the experiments on all datasets
# dataset_names = ["rcv1", "20newsgroups"]
methods = ["implicit_forward", "forward", 'grid_search']
tolerance_decreases = ["constant"]
tols = [1e-5]
n_outers = [50]


dict_n_outers = {}
dict_n_outers["20newsgroups", "implicit_forward"] = 75
dict_n_outers["20newsgroups", "forward"] = 75
dict_n_outers["20newsgroups", "implicit"] = 75
dict_n_outers["20newsgroups", "bayesian"] = 75
dict_n_outers["20newsgroups", "random"] = 75

dict_n_outers["finance", "implicit_forward"] = 125
dict_n_outers["finance", "forward"] = 75
dict_n_outers["finance", "implicit"] = 6
dict_n_outers["finance", "bayesian"] = 75
dict_n_outers["finance", "random"] = 50

dict_tmax = {}
dict_tmax["rcv1"] = 10
dict_tmax["20newsgroups"] = 200
dict_tmax["finance"] = 500


def parallel_function(
        dataset_name, method, tol=1e-5, n_outer=50,
        tolerance_decrease='constant'):
    t_max = dict_tmax[dataset_name]
    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(dataset_name)
    n_samples, n_features = X_train.shape
    # compute alpha_max
    alpha_max = np.abs(X_train.T @ y_train).max() / n_samples
    log_alpha0 = np.log(0.1 * alpha_max)

    idx_nz = scipy.sparse.linalg.norm(X_train, axis=0) != 0
    L_min = scipy.sparse.linalg.norm(
        X_train[:, idx_nz], axis=0).min() ** 2 / n_samples

    log_alpha0_mcp = np.array([log_alpha0, np.log(2 / L_min)])

    list_log_alphas = np.log(alpha_max * np.geomspace(1, 0.0001, 100))
    list_log_gammas = np.log(np.geomspace(1.1 / L_min, 1000 / L_min, 5))

    try:
        n_outer = dict_n_outers[dataset_name, method]
    except:
        n_outer = 50

    if dataset_name == "rcv1":
        size_loop = 2
    else:
        size_loop = 1
    for i in range(size_loop):
        monitor = Monitor()
        warm_start = WarmStart()

        if method == 'grid_search':
            # n_alpha = 100
            # p_alphas = np.geomspace(1, 0.0001, n_alpha)
            grid_searchMCP(
                    X_train, y_train, list_log_alphas, list_log_gammas,
                    X_val, y_val, X_test, y_test, tol, monitor=monitor)
        elif method in ("bayesian", "random"):
            monitor = hyperopt_lasso(
                X_train, y_train, log_alpha0, X_val, y_val, X_test, y_test,
                tol, max_evals=n_outer, method=method)
        else:
            # do line search to find the optimal lambda
            log_alpha, val, grad = grad_search(
                X_train, y_train, log_alpha0_mcp, X_val, y_val, X_test, y_test,
                tol, monitor, method=method, maxit=10000, n_outer=n_outer,
                warm_start=warm_start, niter_jac=100, model="mcp", t_max=t_max)
            del log_alpha, val, grad  # as not used

    monitor.times = np.array(monitor.times)
    monitor.objs = np.array(monitor.objs)
    monitor.objs_test = np.array(monitor.objs_test)
    monitor.log_alphas = np.array(monitor.log_alphas)
    return (dataset_name, method, tol, n_outer, tolerance_decrease,
            monitor.times, monitor.objs, monitor.objs_test,
            monitor.log_alphas, norm(y_val), norm(y_test))


# print("enter sequential")
# for dataset_name, method, n_outer, tolerance_decrease in product(
#         dataset_names, methods, n_outers, tolerance_decreases):
#     results = parallel_function(
#         dataset_name, method, n_outer=n_outer,
#         tolerance_decrease=tolerance_decrease)
#     df = pandas.DataFrame([results])
#     df.columns = [
#         'dataset', 'method', 'tol', 'n_outer', 'tolerance_decrease',
#         'times', 'objs', 'objs_test', 'log_alphas', 'norm y_val',
#         'norm y_test']
#     df.to_pickle("%s_mcp.pkl" % dataset_name)
# print('OK finished sequential')


print("enter sequential")
backend = 'loky'
n_jobs = 1
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(parallel_function)(
        dataset_name, method, n_outer=n_outer,
        tolerance_decrease=tolerance_decrease)
    for dataset_name, method, n_outer,
    tolerance_decrease in product(
        dataset_names, methods, n_outers, tolerance_decreases))
print('OK finished sequential')

# print("enter parallel")
# backend = 'loky'
# n_jobs = len(dataset_names) * len(methods) * len(tolerance_decreases)
# results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
#     delayed(parallel_function)(
#         dataset_name, method, n_outer=n_outer,
#         tolerance_decrease=tolerance_decrease)
#     for dataset_name, method, n_outer,
#     tolerance_decrease in product(
#         dataset_names, methods, n_outers, tolerance_decreases))
# print('OK finished parallel')

df = pandas.DataFrame(results)
df.columns = [
    'dataset', 'method', 'tol', 'n_outer', 'tolerance_decrease',
    'times', 'objs', 'objs_test', 'log_alphas', 'norm y_val',
    'norm y_test']

for dataset_name in dataset_names:
    df[df['dataset'] == dataset_name].to_pickle(
        "%s_mcp.pkl" % dataset_name)
