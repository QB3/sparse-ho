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

from sparse_ho.model.lasso import Lasso
from sparse_ho.model.sparselogreg import SparseLogreg
from sparse_ho.criterion.hout_mse import HeldOutMSE
from sparse_ho.criterion.hout_logistic import HeldOutLogistic
from sparse_ho.utils import Monitor

from sparse_ho.algo.forward import Forward
from sparse_ho.algo.implicit_forward import ImplicitForward
from sparse_ho.algo.implicit import Implicit
from sparse_ho.grid_search import grid_search
from sparse_ho.hyperopt_wrapper import hyperopt_wrapper
# from sparse_ho.bayesian import hyperopt_lasso

from sparse_ho.ho import grad_search

model_name = "lasso"
# model_name = "logreg"

dict_t_max = {}
dict_t_max["rcv1"] = 50
dict_t_max["real-sim"] = 100
dict_t_max["leukemia"] = 10
dict_t_max["20newsgroups"] = 300

#######################################################################
# dataset_names = ["rcv1", "real-sim", "20newsgroups"]
# dataset_names = ["real-sim"]
dataset_names = ["20newsgroups"]
# dataset_names = ["leukemia"]
# uncomment the following line to launch the experiments on other
# datasets:
# dataset_names = ["rcv1", "20newsgroups", "finance"]
methods = [
    "implicit_forward",
    # "implicit",
    "forward", 'grid_search', "bayesian", "random"]
# tolerance_decreases = ["exponential"]
tolerance_decreases = ["constant"]
tols = [1e-7]
n_outers = [75]
n_alphas = 100

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
dict_algo["random"] = Forward
dict_algo["bayesian"] = Forward
dict_algo["grid_search"] = Forward

#######################################################################
# n_jobs = 1
n_jobs = len(dataset_names) * len(methods) * len(tolerance_decreases)
n_jobs = min(n_jobs, 1)
#######################################################################


def parallel_function(
        dataset_name, method, tol=1e-5, n_outer=50,
        tolerance_decrease='constant'):

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(dataset_name)
    n_samples, _ = X_train.shape
    # compute alpha_max
    alpha_max = np.abs(X_train.T @ y_train).max() / n_samples

    if model_name == "logreg":
        alpha_max /= 2
    alpha_min = alpha_max / 10_000
    log_alpha_max = np.log(alpha_max)
    log_alpha_min = np.log(alpha_min)
    log_alpha0 = np.log(0.1 * alpha_max)

    if model_name == "lasso":
        model = Lasso(X_train, y_train)
    elif model_name == "logreg":
        model = SparseLogreg(X_train, y_train)

    try:
        n_outer = dict_n_outers[dataset_name, method]
    except Exception:
        n_outer = 20

    size_loop = 2

    for _ in range(size_loop):
        if model_name == "lasso":
            criterion = HeldOutMSE(X_val, y_val, model, X_test=X_test,
                                   y_test=y_test)
        elif model_name == "logreg":
            criterion = HeldOutLogistic(
                X_val, y_val, model, X_test=X_test, y_test=y_test)
        algo = dict_algo[method](criterion)
        monitor = Monitor()
        if method == 'grid_search':
            log_alphas = np.log(np.geomspace(alpha_max, alpha_min, num=100))
            grid_search(
                algo, None, None, monitor, log_alphas=log_alphas,
                tol=tol)
        elif method == 'random':
            grid_search(
                algo, log_alpha_max, log_alpha_min, monitor, tol=tol, max_evals=n_alphas, t_max=dict_t_max[dataset_name])
        elif method in ("bayesian"):
            hyperopt_wrapper(
                algo, log_alpha_min, log_alpha_max, monitor,
                max_evals=n_alphas, tol=tol, method='bayesian',
                t_max=dict_t_max[dataset_name])
        else:
            # do line search to find the optimal lambda
            grad_search(
                algo, log_alpha0, monitor, n_outer=n_outer, tol=tol,
                tolerance_decrease=tolerance_decrease,
                t_max=dict_t_max[dataset_name])

        monitor.times = np.array(monitor.times)
        monitor.objs = np.array(monitor.objs)
        monitor.objs_test = np.array(monitor.objs_test)
        monitor.log_alphas = np.array(monitor.log_alphas)
    return (dataset_name, method, tol, n_outer, tolerance_decrease,
            monitor.times, monitor.objs, monitor.objs_test,
            monitor.log_alphas, norm(y_val), norm(y_test), log_alpha_max,
            model_name)


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
    'norm y_test', 'log_alpha_max', 'model_name']

for dataset_name in dataset_names:
    df[df['dataset'] == dataset_name].to_pickle(
        "%s_%s.pkl" % (model_name, dataset_name))
