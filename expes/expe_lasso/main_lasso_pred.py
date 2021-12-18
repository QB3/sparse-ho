"""
This is the file to reproduce the experiments of Figure 2:
'Computation time for the HO of the Lasso on real data.'
It is recommended to run this script on a cluster with several CPUs.
"""

from collections import defaultdict
import numpy as np
from joblib import Parallel, delayed, parallel_backend
from itertools import product
import pandas as pd
import celer

from sklearn.model_selection import KFold

from libsvmdata import fetch_libsvm

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE, CrossVal
from sparse_ho.utils import Monitor
from sparse_ho.optimizers import GradientDescent

from sparse_ho import Implicit
from sparse_ho.grid_search import grid_search
from sparse_ho.ho import hyperopt_wrapper

from sparse_ho.ho import grad_search

model_name = "lasso"

dict_t_max = {}
dict_t_max["rcv1_train"] = 120
# dict_t_max["rcv1_train"] = 300
dict_t_max["real-sim"] = 240
dict_t_max["leukemia"] = 10
dict_t_max["news20"] = 1_500

#######################################################################
# uncomment the following line to launch the experiments on other
# dataset_names = ["rcv1_train"]
dataset_names = ["news20"]
methods = ["implicit", "implicit_approx"]
# methods = ["implicit", "implicit_approx"]
# methods = ["implicit"]
tolerance_decreases = ["constant"]
tol = 1e-8
n_outers = [75]

dict_n_outers = defaultdict(lambda: 20, key=None)
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
    # compute alpha_max
    alpha_max = np.abs(X.T @ y).max() / len(y)
    alpha_min = alpha_max / 10_000

    estimator = celer.Lasso(
        fit_intercept=False, max_iter=100, warm_start=True, tol=tol)
    model = Lasso(estimator=estimator)
    n_outer = dict_n_outers[dataset_name, method]

    for t_max in [10, dict_t_max[dataset_name]]:
        sub_criterion = HeldOutMSE(None, None)
        kf = KFold(n_splits=5, shuffle=True, random_state=42)
        criterion = CrossVal(sub_criterion, cv=kf)

        algo = Implicit(tol_lin_sys=1e-3)
        monitor = Monitor()
        if method == 'grid_search':
            grid_search(
                criterion, model, X, y, alpha_min, alpha_max,
                monitor, max_evals=100, tol=tol)
        elif method == 'random' or method == 'bayesian':
            hyperopt_wrapper(
                criterion, model, X, y, alpha_min, alpha_max,
                monitor, max_evals=30, tol=tol, method=method, size_space=1,
                t_max=t_max)
        elif method.startswith("implicit"):
            # do gradient descent to find the optimal lambda
            alpha0 = alpha_max / 100
            n_outer = 30
            if method == 'implicit':
                optimizer = GradientDescent(
                    n_outer=n_outer, p_grad_norm=1, verbose=True, tol=tol,
                    t_max=t_max)
            else:
                optimizer = GradientDescent(
                    n_outer=n_outer, p_grad_norm=1, verbose=True, tol=tol,
                    t_max=t_max,
                    tol_decrease="geom")
            grad_search(
                algo, criterion, model, optimizer, X, y, alpha0,
                monitor)
        else:
            raise NotImplementedError

    monitor.times = np.array(monitor.times)
    monitor.objs = np.array(monitor.objs)
    monitor.objs_test = 0  # TODO
    monitor.alphas = np.array(monitor.alphas)
    results = (
        dataset_name, method, tol, n_outer, tolerance_decrease,
        monitor.times, monitor.objs, monitor.objs_test,
        monitor.alphas, alpha_max,
        model_name)
    df = pd.DataFrame(results).transpose()
    df.columns = [
        'dataset', 'method', 'tol', 'n_outer', 'tolerance_decrease',
        'times', 'objs', 'objs_test', 'alphas', 'alpha_max', 'model_name']
    str_results = "results/%s_%s_%s.pkl" % (
        model_name, dataset_name, method)
    df.to_pickle(str_results)


print("enter parallel")

with parallel_backend("loky", inner_max_num_threads=1):
    Parallel(n_jobs=n_jobs, verbose=100)(
        delayed(parallel_function)(
            dataset_name, method, n_outer=n_outer,
            tolerance_decrease=tolerance_decrease, tol=tol)
        for dataset_name, method, n_outer,
        tolerance_decrease in product(
            dataset_names, methods, n_outers, tolerance_decreases))
print('OK finished parallel')

# df = pd.DataFrame(results)


# for dataset_name in dataset_names:
#     df[df['dataset'] == dataset_name].to_pickle(
#         "results/%s_%s.pkl" % (model_name, dataset_name))
