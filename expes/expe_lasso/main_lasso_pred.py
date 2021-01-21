"""
This is the file to reproduce the experiments of Figure 2:
'Computation time for the HO of the Lasso on real data.'
It is recommended to run this script on a cluster with several CPUs.
"""

import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed, parallel_backend
from itertools import product
import pandas as pd
import celer

from sklearn.model_selection import KFold

from libsvmdata import fetch_libsvm

from sparse_ho.models import Lasso, SparseLogreg
from sparse_ho.criterion import HeldOutMSE, HeldOutLogistic, CrossVal
from sparse_ho.utils import Monitor
from sparse_ho.optimizers import GradientDescent

from sparse_ho import ImplicitForward
from sparse_ho.algo import Implicit
from sparse_ho.grid_search import grid_search
from sparse_ho.ho import hyperopt_wrapper
# from sparse_ho.bayesian import hyperopt_lasso

from sparse_ho.ho import grad_search

model_name = "lasso"
# model_name = "logreg"

dict_t_max = {}
dict_t_max["rcv1"] = 50
dict_t_max["real-sim"] = 100
dict_t_max["leukemia"] = 10
dict_t_max["news20"] = 10_000

#######################################################################
# dataset_names = ["rcv1_train"]
dataset_names = ["news20"]
# dataset_names = ["news20"]
# dataset_names = ["leukemia"]
# uncomment the following line to launch the experiments on other
# datasets:
# dataset_names = ["rcv1", "news20", "finance"]
methods = [
    "implicit_forward", 'grid_search', 'random', 'bayesian']
# tolerance_decreases = ["exponential"]
tolerance_decreases = ["constant"]
# tols = [1e-8]
tol = 1e-8
n_outers = [75]
n_alphas = 100

dict_n_outers = {}
dict_n_outers["news20", "implicit_forward"] = 50
dict_n_outers["news20", "forward"] = 60
dict_n_outers["news20", "implicit"] = 6
dict_n_outers["news20", "bayesian"] = 75
dict_n_outers["news20", "random"] = 35

dict_n_outers["finance", "implicit_forward"] = 125
dict_n_outers["finance", "forward"] = 75
dict_n_outers["finance", "implicit"] = 6
dict_n_outers["finance", "bayesian"] = 75
dict_n_outers["finance", "random"] = 50

# dict_algo = {}
# dict_algo["implicit_forward"] = ImplicitForward
# dict_algo["implicit"] = Implicit
# # dict_algo["forward"] = Forward
# # dict_algo["random"] = Forward
# # dict_algo["bayesian"] = Forward
# # dict_algo["grid_search"] = Forward

#######################################################################
# n_jobs = 1
n_jobs = len(dataset_names) * len(methods) * len(tolerance_decreases)
n_jobs = min(n_jobs, 10)
#######################################################################


def parallel_function(
        dataset_name, method, tol=1e-5, n_outer=50,
        tolerance_decrease='constant'):

    # load data
    X, y = fetch_libsvm(dataset_name)
    y -= y.mean()
    y /= norm(y)
    # compute alpha_max
    alpha_max = np.abs(X.T @ y).max() / len(y)

    if model_name == "logreg":
        alpha_max /= 2
    alpha_min = alpha_max / 10_000
    log_alpha_max = np.log(alpha_max)
    log_alpha_min = np.log(alpha_min)

    if model_name == "lasso":
        estimator = celer.Lasso(
            fit_intercept=False, max_iter=100, warm_start=True, tol=tol)
        model = Lasso(estimator=estimator)
    elif model_name == "logreg":
        model = SparseLogreg(estimator=estimator)

    # TODO improve this
    try:
        n_outer = dict_n_outers[dataset_name, method]
    except Exception:
        n_outer = 20

    size_loop = 2

    for _ in range(size_loop):
        if model_name == "lasso":
            sub_criterion = HeldOutMSE(None, None)
        elif model_name == "logreg":
            criterion = HeldOutLogistic(None, None)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        criterion = CrossVal(sub_criterion, cv=kf)

        algo = ImplicitForward(tol_jac=1e-3)
        monitor = Monitor()
        t_max = dict_t_max[dataset_name]
        if method == 'grid_search':
            grid_search(
                algo, criterion, model, X, y, log_alpha_min, log_alpha_max,
                monitor, max_evals=100, tol=tol, t_max=t_max)
        elif method == 'random' or method == 'bayesian':
            hyperopt_wrapper(
                algo, criterion, model, X, y, log_alpha_min, log_alpha_max,
                monitor, max_evals=30, tol=tol, method=method, size_space=1,
                t_max=t_max)
        elif method == "implicit_forward":
            # do gradient descent to find the optimal lambda
            log_alpha0 = np.log(alpha_max / 30)
            optimizer = GradientDescent(
                n_outer=30, p_grad0=1, verbose=True, tol=tol, t_max=t_max)
            grad_search(
                algo, criterion, model, optimizer, X, y, log_alpha0,
                monitor)
        else:
            1 / 0

    monitor.times = np.array(monitor.times)
    monitor.objs = np.array(monitor.objs)
    monitor.objs_test = 0  # TODO
    monitor.log_alphas = np.array(monitor.log_alphas)
    return (dataset_name, method, tol, n_outer, tolerance_decrease,
            monitor.times, monitor.objs, monitor.objs_test,
            monitor.log_alphas, log_alpha_max,
            model_name)


print("enter sequential")
backend = 'loky'

with parallel_backend("loky", inner_max_num_threads=1):
    results = Parallel(n_jobs=n_jobs, verbose=100)(
        delayed(parallel_function)(
            dataset_name, method, n_outer=n_outer,
            tolerance_decrease=tolerance_decrease, tol=tol)
        for dataset_name, method, n_outer,
        tolerance_decrease in product(
            dataset_names, methods, n_outers, tolerance_decreases))
    print('OK finished parallel')

df = pd.DataFrame(results)
df.columns = [
    'dataset', 'method', 'tol', 'n_outer', 'tolerance_decrease',
    'times', 'objs', 'objs_test', 'log_alphas', 'log_alpha_max', 'model_name']

for dataset_name in dataset_names:
    df[df['dataset'] == dataset_name].to_pickle(
        "results/%s_%s.pkl" % (model_name, dataset_name))
