import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
import pandas
from bcdsugar.utils import Monitor
from sparse_ho.ho import grad_search
from itertools import product
from sparse_ho.criterion import Logistic
from sparse_ho.models import SparseLogregGradSearch
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.implicit import Implicit
from sparse_ho.datasets.real import get_data
from sparse_ho.grid_search import grid_search


dataset_names = ["rcv1"]
# dataset_names = ["real-sim"]
# dataset_names = ["20news"]

# methods = ["grid_search"]
methods = ["implicit_forward", "forward",
           "grid_search", "random"]
# "grid_search",
tolerance_decreases = ["constant"]
tols = 1e-7
n_outers = [25]

dict_t_max = {}
dict_t_max["rcv1"] = 50
dict_t_max["real-sim"] = 100
dict_t_max["leukemia"] = 10
dict_t_max["20news"] = 500


def parallel_function(
        dataset_name, method, tol=1e-5, n_outer=50,
        tolerance_decrease='exponential'):

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(dataset_name)
    n_samples, n_features = X_train.shape
    print('n_samples', n_samples)
    print('n_features', n_features)
    y_train[y_train == 0.0] = -1.0
    y_val[y_val == 0.0] = -1.0
    y_test[y_test == 0.0] = -1.0

    alpha_max = np.max(np.abs(X_train.T @ y_train))
    alpha_max /= X_train.shape[0]
    alpha_max /= 4
    log_alpha_max = np.log(alpha_max)

    alpha_min = alpha_max * 1e-4
    alphas = np.geomspace(alpha_max, alpha_min, 10)
    log_alphas = np.log(alphas)

    log_alpha0 = np.log(0.1 * alpha_max)
    log_alpha_max = np.log(alpha_max)
    n_outer = 25

    if dataset_name == "rcv1":
        size_loop = 2
    else:
        size_loop = 2
    model = SparseLogreg(
        X_train, y_train, max_iter=1000, log_alpha_max=log_alpha_max)
    for i in range(size_loop):
        monitor = Monitor()

        if method == "implicit_forward":
            criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = ImplicitForward(tol_jac=1e-5, n_iter_jac=100)
            grad_search(
                algo=algo, criterion=criterion, verbose=False,
                log_alpha0=log_alpha0, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "forward":
            criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward(criterion)
            grad_search(
                algo=algo, criterion=criterion,
                log_alpha0=log_alpha0, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "implicit":
            criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Implicit()
            grad_search(
                algo=algo, criterion=criterion,
                log_alpha0=log_alpha0, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "grid_search":
            criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward()
            # log_alpha_min = np.log(alpha_min)
            log_alphas = np.log(np.geomspace(alpha_max, alpha_min, num=100))
            log_alpha_opt, min_g_func = grid_search(
                algo, criterion, None, None, monitor, tol=tol, samp="grid",
                t_max=dict_t_max[dataset_name], log_alphas=log_alphas)

        elif method == "random":
            criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward()
            log_alpha_min = np.log(alpha_min)
            log_alpha_opt, min_g_func = grid_search(
                algo, criterion, log_alpha_min, np.log(alpha_max), monitor,
                max_evals=100, tol=tol, samp="random",
                t_max=dict_t_max[dataset_name])
            print(log_alpha_opt)

        elif method == "lhs":
            criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward()
            log_alpha_min = np.log(alpha_min)
            log_alpha_opt, min_g_func = grid_search(
                algo, criterion, log_alpha_min, np.log(alpha_max), monitor,
                max_evals=100, tol=tol, samp="lhs",
                t_max=dict_t_max[dataset_name])
            print(log_alpha_opt)

    monitor.times = np.array(monitor.times).copy()
    monitor.objs = np.array(monitor.objs).copy()
    monitor.objs_test = np.array(monitor.objs_test).copy()
    monitor.log_alphas = np.array(monitor.log_alphas).copy()
    return (dataset_name, method, tol, n_outer, tolerance_decrease,
            monitor.times, monitor.objs, monitor.objs_test,
            monitor.log_alphas, norm(y_val), norm(y_test), log_alpha_max)


print("enter parallel")
backend = 'loky'
# n_jobs = 1
n_jobs = len(methods)
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
    'norm y_test', "log_alpha_max"]

for dataset_name in dataset_names:
    df[df['dataset'] == dataset_name].to_pickle(
        "%s.pkl" % dataset_name)
