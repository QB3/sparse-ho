import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
import pandas
from bcdsugar.utils import Monitor
from sparse_ho.ho import grad_search
from itertools import product
from sparse_ho.criterion import HeldOutSmoothedHinge
from sparse_ho.models import SVM
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.implicit import Implicit
from sparse_ho.datasets.real import get_data
from sparse_ho.grid_search import grid_search

# from my_data import get_data

dataset_names = ["real-sim"]

# methods = ["implicit_forward", "implicit"]
methods = ["forward", "implicit_forward"]
# "grid_search",
tolerance_decreases = ["constant"]
tols = 1e-5
n_outers = [1]

dict_t_max = {}
dict_t_max["rcv1"] = 50
dict_t_max["real-sim"] = 100
dict_t_max["leukemia"] = 10
dict_t_max["20news"] = 500


def parallel_function(
        dataset_name, method, tol=1e-5, n_outer=50,
        tolerance_decrease='exponential'):

    # load data
    X_train, X_val, X_test, y_train, y_val, y_test = get_data(
        dataset_name, csr=True)
    n_samples, n_features = X_train.shape
    print('n_samples', n_samples)
    print('n_features', n_features)
    y_train[y_train == 0.0] = -1.0
    y_val[y_val == 0.0] = -1.0
    y_test[y_test == 0.0] = -1.0

    C_max = 100
    logC = np.log(1e-2)
    n_outer = 5

    if dataset_name == "rcv1":
        size_loop = 1
    else:
        size_loop = 1
    model = SVM(
        X_train, y_train, logC, max_iter=10000, tol=tol)
    for i in range(size_loop):
        monitor = Monitor()

        if method == "implicit_forward":
            criterion = HeldOutSmoothedHinge(
                X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = ImplicitForward(criterion, tol_jac=1e-3, n_iter_jac=100)
            _, _, _ = grad_search(
                algo=algo, verbose=False,
                log_alpha0=logC, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "forward":
            criterion = HeldOutSmoothedHinge(
                X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward(criterion)
            _, _, _ = grad_search(
                algo=algo,
                log_alpha0=logC, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "implicit":
            criterion = HeldOutSmoothedHinge(
                X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Implicit(criterion)
            _, _, _ = grad_search(
                algo=algo,
                log_alpha0=logC, tol=tol,
                n_outer=n_outer, monitor=monitor,
                t_max=dict_t_max[dataset_name],
                tolerance_decrease=tolerance_decrease)

        elif method == "grid_search":
            criterion = HeldOutSmoothedHinge(
                X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward(criterion)
            log_alpha_min = np.log(1e-2)
            log_alpha_opt, min_g_func = grid_search(
                algo, log_alpha_min, np.log(C_max), monitor, max_evals=25,
                tol=tol, samp="grid")
            print(log_alpha_opt)

        elif method == "random":
            criterion = HeldOutSmoothedHinge(
                X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward(criterion)
            log_alpha_min = np.log(1e-2)
            log_alpha_opt, min_g_func = grid_search(
                algo, log_alpha_min, np.log(C_max), monitor, max_evals=25,
                tol=tol, samp="random")
            print(log_alpha_opt)

        elif method == "lhs":
            criterion = HeldOutSmoothedHinge(
                X_val, y_val, model, X_test=X_test, y_test=y_test)
            algo = Forward(criterion)
            log_alpha_min = np.log(1e-2)
            log_alpha_opt, min_g_func = grid_search(
                algo, log_alpha_min, np.log(C_max), monitor, max_evals=25,
                tol=tol, samp="lhs")
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
