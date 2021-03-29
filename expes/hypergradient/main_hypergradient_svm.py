import os
import numpy as np
import pandas
from itertools import product
from sklearn.model_selection import StratifiedShuffleSplit
from joblib import Parallel, delayed, parallel_backend
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm

from lightning.classification import LinearSVC
from libsvmdata import fetch_libsvm

from sparse_ho.models import SVM
from sparse_ho.criterion import HeldOutSmoothedHinge
from sparse_ho import ImplicitForward, Implicit
from sparse_ho import Forward
from sparse_ho.utils import Monitor


tol = 1e-32
methods = ["ground_truth", "forward", "implicit_forward", "sota"]
# div_alphas = [100]
dataset_names = ["rcv1_train"]
# dataset_names = ["real-sim"]

n_points = 10
dict_max_iter = {}
dict_max_iter["real-sim"] = np.linspace(5, 100, n_points, dtype=np.int)
dict_max_iter["rcv1_train"] = np.linspace(5, 5_000, n_points, dtype=np.int)

dict_logC = {}
dict_logC["real-sim"] = [np.log(0.1)]
dict_logC["rcv1_train"] = [np.log(0.2)]


def parallel_function(
        dataset_name, method):
    X, y = fetch_libsvm(dataset_name)
    X, y = fetch_libsvm(dataset_name)
    if dataset_name == "real-sim":
        X = X[:, :2000]
    X = csr_matrix(X)  # very important for SVM
    my_bool = norm(X, axis=1) != 0
    X = X[my_bool, :]
    y = y[my_bool]
    logC = dict_logC[dataset_name]
    for max_iter in dict_max_iter[dataset_name]:
        print("Dataset %s, max iter %i" % (method, max_iter))
        for i in range(2):  # TODO change this
            sss1 = StratifiedShuffleSplit(
                n_splits=2, test_size=0.3333, random_state=0)
            idx_train, idx_val = sss1.split(X, y)
            idx_train = idx_train[0]
            idx_val = idx_val[0]

            monitor = Monitor()
            criterion = HeldOutSmoothedHinge(idx_train, idx_val)
            model = SVM(estimator=None, max_iter=10_000)

            if method == "ground_truth":
                for file in os.listdir("results_svm/"):
                    if file.startswith("hypergradient_svm_%s_%s" % (
                            dataset_name, method)):
                        return
                clf = LinearSVC(
                        C=np.exp(logC), tol=1e-32, max_iter=10_000,
                        loss='hinge', permute=False)
                algo = Implicit(criterion)
                model.estimator = clf
                val, grad = criterion.get_val_grad(
                        model, X, y, logC, algo.compute_beta_grad, tol=1e-14,
                        monitor=monitor)
            else:
                if method == "sota":
                    clf = LinearSVC(
                        C=np.exp(logC), loss='hinge', max_iter=max_iter,
                        tol=1e-32, permute=False)
                    model.estimator = clf
                    algo = ImplicitForward(
                        tol_jac=1e-32, n_iter_jac=max_iter,
                        use_stop_crit=False)
                elif method == "forward":
                    algo = Forward(use_stop_crit=False)
                elif method == "implicit_forward":
                    algo = ImplicitForward(
                        tol_jac=1e-8, n_iter_jac=max_iter, use_stop_crit=False)
                else:
                    raise NotImplementedError
                algo.max_iter = max_iter
                algo.use_stop_crit = False
                val, grad = criterion.get_val_grad(
                        model, X, y, logC, algo.compute_beta_grad, tol=tol,
                        monitor=monitor, max_iter=max_iter)

        results = (
            dataset_name, method, max_iter,
            val, grad, monitor.times[0])
        df = pandas.DataFrame(results).transpose()
        df.columns = [
            'dataset', 'method', 'maxit', 'val', 'grad', 'time']
        str_results = "results_svm/hypergradient_svm_%s_%s_%i.pkl" % (
            dataset_name, method, max_iter)
        df.to_pickle(str_results)


if __name__ == "__main__":
    print("enter parallel")
    backend = 'loky'
    n_jobs = 1
    with parallel_backend(backend, n_jobs=n_jobs, inner_max_num_threads=1):
        Parallel()(
            delayed(parallel_function)(dataset_name, method)
            for dataset_name, method in product(dataset_names, methods))
    print('OK finished parallel')
