import os
from itertools import product
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import pandas
from sklearn.model_selection import StratifiedShuffleSplit, KFold

from celer import LogisticRegression
from libsvmdata import fetch_libsvm

from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import HeldOutLogistic, CrossVal
from sparse_ho import ImplicitForward, Implicit
from sparse_ho import Forward, Backward
from sparse_ho.utils import Monitor


tol = 1e-6
methods = ["forward", "implicit_forward", "celer", "ground_truth"]
# methods = ["ground_truth"]
div_alphas = [3, 10]
# dataset_names = ["sector_train"]
# dataset_names = ["rcv1_train", "real-sim", "news20"]
dataset_names = ["real-sim"]
rep = 10

dict_maxits = {}
dict_maxits[("leukemia", 10)] = np.linspace(5, 50, rep, dtype=np.int)
dict_maxits[("leukemia", 100)] = np.linspace(5, 300, rep, dtype=np.int)
dict_maxits[("real-sim", 10)] = np.linspace(5, 50, rep, dtype=np.int)
dict_maxits[("real-sim", 3)] = np.linspace(5, 50, rep, dtype=np.int)
dict_maxits[("rcv1_train", 10)] = np.linspace(5, 100, rep, dtype=np.int)
dict_maxits[("rcv1_train", 3)] = np.linspace(5, 100, rep, dtype=np.int)
dict_maxits[("news20", 10)] = np.linspace(5, 1000, rep, dtype=np.int)
dict_maxits[("news20", 100)] = np.linspace(5, 10000, rep, dtype=np.int)


def parallel_function(
        dataset_name, div_alpha, method, ind_rep, random_state=10):
    maxit = dict_maxits[(dataset_name, div_alpha)][ind_rep]
    print("Dataset %s, algo %s, maxit %i" % (dataset_name, method, maxit))
    X, y = fetch_libsvm(dataset_name)
    n_samples = len(y)

    kf = KFold(n_splits=5, random_state=random_state, shuffle=True)

    for i in range(2):
        alpha_max = np.max(np.abs(X.T @ y))
        alpha_max /= (4 * X.shape[0])
        log_alpha = np.log(alpha_max / div_alpha)
        alpha = np.exp(log_alpha)
        monitor = Monitor()
        if method == "celer":
            clf = LogisticRegression(
                C=(1 / (alpha * X.shape[0])),
                solver="celer",
                tol=tol, max_iter=maxit)
            model = SparseLogreg(estimator=clf, max_iter=maxit)
            criterion = HeldOutLogistic(None, None)
            cross_val = CrossVal(cv=kf, criterion=criterion)
            algo = ImplicitForward(
                tol_jac=tol, n_iter_jac=maxit, use_stop_crit=False)
            algo.max_iter = maxit
            val, grad = cross_val.get_val_grad(
                    model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol,
                    monitor=monitor, max_iter=maxit)
        elif method == "ground_truth":
            for file in os.listdir("results_logreg/"):
                if file.startswith(
                    "hypergradient_%s_%i_%s" % (
                        dataset_name, div_alpha, method)):
                    return
                else:
                    clf = LogisticRegression(
                            C=(1 / (alpha * X.shape[0])),
                            solver="celer",
                            tol=1e-8, max_iter=maxit)
                    criterion = HeldOutLogistic(None, None)
                    cross_val = CrossVal(cv=kf, criterion=criterion)
                    algo = Implicit(criterion)
                    model = SparseLogreg(estimator=clf, max_iter=10000)
                    val, grad = cross_val.get_val_grad(
                        model, X, y, log_alpha, algo.get_beta_jac_v, tol=1e-8,
                        monitor=monitor)
        else:
            model = SparseLogreg(max_iter=maxit)
            criterion = HeldOutLogistic(None, None)
            cross_val = CrossVal(cv=kf, criterion=criterion)
            if method == "forward":
                algo = Forward(use_stop_crit=False)
            elif method == "implicit_forward":
                algo = ImplicitForward(use_stop_crit=False,
                    tol_jac=tol, n_iter_jac=maxit, max_iter=1000)
            elif method == "implicit":
                algo = Implicit(use_stop_crit=False, max_iter=1000)
            elif method == "backward":
                algo = Backward()
            else:
                1 / 0
            algo.max_iter = maxit
            algo.use_stop_crit = False
            val, grad = cross_val.get_val_grad(
                    model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol,
                    monitor=monitor, max_iter=maxit)

        results = (
            dataset_name, div_alpha, method, maxit,
            val, grad, monitor.times[0])
    df = pandas.DataFrame(results).transpose()
    df.columns = [
        'dataset', 'div_alpha', 'method', 'maxit', 'val', 'grad',
        'time']
    str_results = "results_logreg/hypergradient_%s_%i_%s_%i.pkl" % (
        dataset_name, div_alpha, method, maxit)
    df.to_pickle(str_results)


print("enter parallel")
backend = 'loky'
n_jobs = 15
# 
with parallel_backend(backend, n_jobs=n_jobs, inner_max_num_threads=1):
    Parallel()(
        delayed(parallel_function)(
            dataset_name, div_alpha, method, ind_rep)
        for dataset_name, div_alpha, method, ind_rep in product(
            dataset_names, div_alphas, methods, range(rep)))
print('OK finished parallel')
