import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
import pandas
from bcdsugar.utils import Monitor
from sparse_ho.ho import grad_search
from itertools import product
from sparse_ho.criterion import Logistic
from sparse_ho.models import SparseLogreg
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.implicit import Implicit
from sparse_ho.datasets.real import get_data
from sparse_ho.grid_search import grid_search

# from my_data import get_data

dataset_names = ["rcv1"]

# methods = ["implicit_forward", "implicit"]
methods = ["implicit", "implicit_forward", "forward", "grid_search"]
# "grid_search",
tolerance_decreases = ["exponential"]
tols = 1e-5
n_outers = [1]

dict_t_max = {}
dict_t_max["rcv1"] = 500
dict_t_max["real-sim"] = 500


def parallel_function(
        dataset_name, method, tol=1e-5, n_outer=50,
        tolerance_decrease='exponential'):

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(dataset_name)
    n_samples, n_features = X_train.shape
    print('n_samples', n_samples)
    print('n_features', n_features)

    alpha_max = np.max(np.abs(X_train.T @ (- y_train)))
    alpha_max /= (2 * n_samples)
    log_alpha0 = np.log(0.2 * alpha_max)
    log_alpha_max = np.log(alpha_max)
    n_outer = 50

    if dataset_name == "rcv1":
        size_loop = 1
    else:
        size_loop = 1
    model = SparseLogreg(
        X_train, y_train, log_alpha0, log_alpha_max, max_iter=1000, tol=tol)
    criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
    for i in range(size_loop):
        monitor = Monitor()

        if method == "implicit_forward":
            algo = ImplicitForward(criterion, tol_jac=1e-3, n_iter_jac=100)
            _, _, _ = grad_search(
                algo=algo, verbose=False,
                log_alpha0=log_alpha0, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "forward":
            algo = Forward(criterion)
            _, _, _ = grad_search(
                algo=algo,
                log_alpha0=log_alpha0, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "implicit":
            algo = Implicit(criterion)
            _, _, _ = grad_search(
                algo=algo,
                log_alpha0=log_alpha0, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "grid_search":
            algo = Forward(criterion)
            log_alpha_min = np.log(1e-8 * alpha_max)
            log_alpha_opt, min_g_func = grid_search(
                algo, log_alpha_min, 0.2 * log_alpha_max, monitor, max_evals=5,
                tol=tol, samp="grid")
            print(log_alpha_opt)

    monitor.times = np.array(monitor.times)
    monitor.objs = np.array(monitor.objs)
    monitor.objs_test = np.array(monitor.objs_test)
    monitor.log_alphas = np.array(monitor.log_alphas)
    return (dataset_name, method, tol, n_outer, tolerance_decrease,
            monitor.times, monitor.objs, monitor.objs_test,
            monitor.log_alphas, norm(y_val), norm(y_test))


print("enter parallel")
backend = 'loky'
n_jobs = 1
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(parallel_function)(
        dataset_name, method, n_outer=n_outer,
        tolerance_decrease=tolerance_decrease, tol=tols)
    for dataset_name, method, n_outer,
    tolerance_decrease in product(
        dataset_names, methods, n_outers, tolerance_decreases))
print('OK finished parallel')

df = pandas.DataFrame(results)
df.columns = [
    'dataset', 'method', 'tol', 'n_outer', 'tolerance_decrease',
    'times', 'objs', 'objs_test', 'log_alphas', 'norm y_val',
    'norm y_test']

for dataset_name in dataset_names:
    df[df['dataset'] == dataset_name].to_pickle(
        "%s.pkl" % dataset_name)
