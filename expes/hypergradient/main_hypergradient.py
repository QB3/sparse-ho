import os
from itertools import product
import numpy as np
from joblib import Parallel, delayed, parallel_backend
import pandas

from celer import Lasso as Lasso_celer
from libsvmdata import fetch_libsvm

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import ImplicitForward, Implicit
from sparse_ho import Forward, Backward
from sparse_ho.utils import Monitor


tol = 1e-32
# methods = ["implicit"]
methods = ["celer"]
div_alphas = [100]
dataset_names = ["news20"]
n_points = 10
dict_maxits = {}
dict_maxits[("real-sim", 10)] = np.linspace(5, 50, n_points, dtype=np.int)
dict_maxits[("real-sim", 100)] = np.linspace(5, 200, n_points, dtype=np.int)
dict_maxits[("rcv1_train", 10)] = np.linspace(5, 150, n_points, dtype=np.int)
dict_maxits[("rcv1_train", 100)] = np.linspace(5, 1000, n_points, dtype=np.int)
dict_maxits[("news20", 10)] = np.linspace(5, 1000, n_points, dtype=np.int)
dict_maxits[("news20", 100)] = np.linspace(5, 2500, n_points, dtype=np.int)


def parallel_function(
        dataset_name, div_alpha, method):
    X, y = fetch_libsvm(dataset_name)
    n_samples = len(y)
    if dataset_name == "news20" and div_alpha == 100:
        rng = np.random.RandomState(42)
        y += rng.randn(n_samples) * 0.01
    for max_iter in dict_maxits[(dataset_name, div_alpha)]:
        print("Dataset %s, max_iter %i" % (method, max_iter))
        for i in range(2):  # TODO to change this
            rng = np.random.RandomState(i)
            idx_train = rng.choice(n_samples, n_samples//2, replace=False)
            idx = np.arange(0, n_samples)
            idx_val = idx[np.logical_not(np.isin(idx, idx_train))]
            alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train])))
            alpha_max /= len(idx_train)
            log_alpha = np.log(alpha_max / div_alpha)
            monitor = Monitor()
            if method == "celer":
                clf = Lasso_celer(
                    alpha=np.exp(log_alpha), fit_intercept=False,
                    tol=1e-12, max_epochs=max_iter)
                model = Lasso(estimator=clf)
                criterion = HeldOutMSE(idx_train, idx_val)
                algo = Implicit(
                    tol_lin_sys=1e-32, max_iter_lin_sys=max_iter)
                # algo = ImplicitForward(
                #     tol_jac=1e-32, n_iter_jac=max_iter, use_stop_crit=False)
                algo.max_iter = max_iter
                val, grad = criterion.get_val_grad(
                        model, X, y, log_alpha, algo.compute_beta_grad,
                        tol=1e-12, monitor=monitor, max_iter=max_iter)
            elif method == "ground_truth":
                for file in os.listdir("results/"):
                    if file.startswith(
                            "hypergradient_%s_%i_%s" % (
                            dataset_name, div_alpha, method)):
                        return
                clf = Lasso_celer(
                        alpha=np.exp(log_alpha), fit_intercept=False,
                        warm_start=True, tol=1e-14)
                criterion = HeldOutMSE(idx_train, idx_val)
                if dataset_name == "news20":
                    algo = ImplicitForward(
                        tol_jac=1e-11, n_iter_jac=100000,
                        max_iter=max_iter)
                else:
                    algo = Implicit(criterion)
                model = Lasso(estimator=clf)
                val, grad = criterion.get_val_grad(
                        model, X, y, log_alpha, algo.compute_beta_grad,
                        tol=1e-14, monitor=monitor)
            else:
                model = Lasso()
                criterion = HeldOutMSE(idx_train, idx_val)
                if method == "forward":
                    algo = Forward(use_stop_crit=False)
                elif method == "implicit_forward":
                    algo = ImplicitForward(
                        tol_jac=1e-8, n_iter_jac=max_iter, use_stop_crit=False,
                        max_iter=max_iter)
                elif method == "implicit":
                    algo = Implicit(
                        max_iter_lin_sys=max_iter, max_iter=max_iter)
                elif method == "backward":
                    algo = Backward()
                else:
                    raise NotImplementedError
                algo.max_iter = max_iter
                algo.use_stop_crit = False
                val, grad = criterion.get_val_grad(
                        model, X, y, log_alpha, algo.compute_beta_grad, tol=tol,
                        monitor=monitor, max_iter=max_iter)

        results = (
            dataset_name, div_alpha, method, max_iter,
            val, grad, monitor.times[0])
        df = pandas.DataFrame(results).transpose()
        df.columns = [
            'dataset', 'div_alpha', 'method', 'max_iter', 'val', 'grad', 'time']
        str_results = "results/hypergradient_%s_%i_%s_%i.pkl" % (
            dataset_name, div_alpha, method, max_iter)
        df.to_pickle(str_results)


print("enter parallel")
backend = 'loky'
n_jobs = 1
with parallel_backend(backend, n_jobs=n_jobs, inner_max_num_threads=1):
    Parallel()(
        delayed(parallel_function)(
            dataset_name, div_alpha, method)
        for dataset_name, div_alpha, method in product(
            dataset_names, div_alphas, methods))
print('OK finished parallel')
