import os
from itertools import product
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import pandas
from sklearn.model_selection import StratifiedShuffleSplit

from celer import Lasso as Lasso_celer
from libsvmdata import fetch_libsvm

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import ImplicitForward, Implicit
from sparse_ho import Forward, Backward
from sparse_ho.utils import Monitor


tol = 1e-16
methods = ["forward", "implicit_forward", "celer", "ground_truth", 'backward']
# methods = ["ground_truth"]
div_alphas = [10]
dataset_names = ["real-sim"]
# dataset_names = ["leukemia", "rcv1_train", "real-sim", "news20"]
rep = 5

dict_maxits = {}
dict_maxits[("real-sim", 10)] = np.linspace(5, 50, rep, dtype=np.int)
dict_maxits[("real-sim", 100)] = np.linspace(5, 200, rep, dtype=np.int)
dict_maxits[("rcv1_train", 10)] = np.linspace(5, 150, rep, dtype=np.int)
dict_maxits[("rcv1_train", 100)] = np.linspace(5, 1000, rep, dtype=np.int)

def parallel_function(
        dataset_name, div_alpha, method, ind_rep, random_state=42):
    maxit = dict_maxits[(dataset_name, div_alpha)][ind_rep]
    print("Dataset %s, algo %s, maxit %i" % (dataset_name, method, maxit))
    X, y = fetch_libsvm(dataset_name)
    n_samples = len(y)
    sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.3333, random_state=0)
    idx_train, idx_val = sss1.split(X, y)
    idx_train = idx_train[0]
    idx_val = idx_val[0]

    for i in range(2):
        alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
        log_alpha = np.log(alpha_max / div_alpha)
        monitor = Monitor()
        if method == "celer":
            clf = Lasso_celer(
                alpha=np.exp(log_alpha), fit_intercept=False,
                # TODO maybe change this tol
                tol=1e-12, max_iter=maxit)
            model = Lasso(estimator=clf, max_iter=maxit)
            criterion = HeldOutMSE(idx_train, idx_val)
            algo = ImplicitForward(
                tol_jac=1e-32, n_iter_jac=maxit, use_stop_crit=False)
            algo.max_iter = maxit
            val, grad = criterion.get_val_grad(
                    model, X, y, log_alpha, algo.get_beta_jac_v, tol=1e-12,
                    monitor=monitor, max_iter=maxit)
        elif method == "ground_truth":
            for file in os.listdir("results/"):
                if file.startswith(
                    "hypergradient_%s_%i_%s" % (
                        dataset_name, div_alpha, method)):
                    return
                else:
                    clf = Lasso_celer(
                            alpha=np.exp(log_alpha), fit_intercept=False,
                            warm_start=True, tol=1e-14, max_iter=10000)
                    criterion = HeldOutMSE(idx_train, idx_val)
                    algo = Implicit(criterion)
                    model = Lasso(estimator=clf, max_iter=10000)
                    val, grad = criterion.get_val_grad(
                        model, X, y, log_alpha, algo.get_beta_jac_v, tol=1e-14,
                        monitor=monitor)
        else:
            model = Lasso(max_iter=maxit)
            criterion = HeldOutMSE(idx_train, idx_val)
            if method == "forward":
                algo = Forward()
            elif method == "implicit_forward":
                algo = ImplicitForward(
                    tol_jac=1e-8, n_iter_jac=maxit, max_iter=1000)
            elif method == "implicit":
                algo = Implicit(max_iter=1000)
            elif method == "backward":
                algo = Backward()
            else:
                1 / 0
            algo.max_iter = maxit
            algo.use_stop_crit = False
            val, grad = criterion.get_val_grad(
                    model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol,
                    monitor=monitor, max_iter=maxit)

        results = (
            dataset_name, div_alpha, method, maxit,
            val, grad, monitor.times[0])
    df = pandas.DataFrame(results).transpose()
    df.columns = [
        'dataset', 'div_alpha', 'method', 'maxit', 'val', 'grad',
        'time']
    str_results = "results/hypergradient_%s_%i_%s_%i.pkl" % (
        dataset_name, div_alpha, method, maxit)
    df.to_pickle(str_results)


print("enter parallel")
backend = 'loky'
n_jobs = 15
with parallel_backend(backend, n_jobs=n_jobs):
    Parallel()(
        delayed(parallel_function)(
            dataset_name, div_alpha, method, ind_rep)
        for dataset_name, div_alpha, method, ind_rep in product(
            dataset_names, div_alphas, methods, range(rep)))
print('OK finished parallel')
