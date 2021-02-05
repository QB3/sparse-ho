from joblib import Parallel, delayed, parallel_backend
from itertools import product
import numpy as np
from sparse_ho.utils import Monitor
from andersoncd.data.real import get_hiva_agnostic
from sparse_ho import Forward, Backward
from sparse_ho.models import Lasso, ElasticNet
from sparse_ho.tests.cvxpylayer import lasso_cvxpy, enet_cvxpy
from sparse_ho.criterion import HeldOutMSE
import time

X, y = get_hiva_agnostic(normalize_y=True)
n_samples, n_features = X.shape
idx_train = np.arange(0, (3 * n_samples) // 4)
idx_val = np.arange(n_samples // 4, n_samples)

name_models = ["lasso", "enet"]

dict_models = {}
dict_models["lasso"] = Lasso()
dict_models["enet"] = ElasticNet()

dict_cvxpy = {}
dict_cvxpy["lasso"] = lasso_cvxpy
dict_cvxpy["enet"] = enet_cvxpy

dict_ncols = {}
dict_ncols[10] = np.geomspace(100, n_features, num=10, dtype=int)
dict_ncols[100] = np.geomspace(100, n_features, num=10, dtype=int)


tol = 1e-6
l1_ratio = 0.8
repeat = 10
div_alphas = [100]


def parallel_function(name_model, div_alpha):
    index_col = np.arange(10)
    alpha_max = (np.abs(X[np.ix_(idx_train, index_col)].T
                 @ y[idx_train])).max() / len(idx_train)
    if name_model == "lasso":
        log_alpha = np.log(alpha_max / div_alpha)
    elif name_model == "enet":
        alpha0 = alpha_max / div_alpha
        alpha1 = (1 - l1_ratio) * alpha0 / l1_ratio
        log_alpha = np.log(np.array([alpha0, alpha1]))

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Forward()
    monitor = Monitor()
    val, grad = criterion.get_val_grad(
        dict_models[name_model], X[:, index_col], y, log_alpha,
        algo.get_beta_jac_v, tol=tol, monitor=monitor)

    criterion = HeldOutMSE(idx_train, idx_val)
    algo = Backward()
    monitor = Monitor()
    val, grad = criterion.get_val_grad(
        dict_models[name_model], X[:, index_col], y, log_alpha,
        algo.get_beta_jac_v, tol=tol, monitor=monitor)

    val_cvxpy, grad_cvxpy = dict_cvxpy[name_model](
        X[:, index_col], y, np.exp(log_alpha), idx_train, idx_val)

    list_times_fwd = []
    list_times_bwd = []
    list_times_cvxpy = []
    for n_col in dict_ncols[div_alpha]:
        temp_fwd = []
        temp_bwd = []
        temp_cvxpy = []
        for i in range(repeat):

            rng = np.random.RandomState(i)
            index_col = rng.choice(n_features, n_col, replace=False)
            alpha_max = (np.abs(X[np.ix_(idx_train, index_col)].T
                         @ y[idx_train])).max() / len(idx_train)
            if name_model == "lasso":
                log_alpha = np.log(alpha_max / div_alpha)
            elif name_model == "enet":
                alpha0 = alpha_max / div_alpha
                alpha1 = (1 - l1_ratio) * alpha0 / l1_ratio
                log_alpha = np.log(np.array([alpha0, alpha1]))

            criterion = HeldOutMSE(idx_train, idx_val)
            algo = Forward()
            monitor = Monitor()
            val, grad = criterion.get_val_grad(
                dict_models[name_model], X[:, index_col], y,
                log_alpha, algo.get_beta_jac_v,
                tol=tol, monitor=monitor)
            temp_fwd.append(monitor.times)

            criterion = HeldOutMSE(idx_train, idx_val)
            algo = Backward()
            monitor = Monitor()
            val, grad = criterion.get_val_grad(
                dict_models[name_model], X[:, index_col], y,
                log_alpha, algo.get_beta_jac_v,
                tol=tol, monitor=monitor)
            temp_bwd.append(monitor.times)

            t0 = time.time()
            val_cvxpy, grad_cvxpy = dict_cvxpy[name_model](
                X[:, index_col], y, np.exp(log_alpha), idx_train, idx_val)
            temp_cvxpy.append(time.time() - t0)

            print(np.abs(grad - grad_cvxpy * np.exp(log_alpha)))
        list_times_fwd.append(np.mean(np.array(temp_fwd)))
        list_times_bwd.append(np.mean(np.array(temp_bwd)))
        list_times_cvxpy.append(np.mean(np.array(temp_cvxpy)))

    np.save("results/times_%s_forward_%s" % (name_model, div_alpha),
            list_times_fwd)
    np.save("results/times_%s_backward_%s" % (name_model, div_alpha),
            list_times_bwd)
    np.save("results/times_%s_cvxpy_%s" % (name_model, div_alpha),
            list_times_cvxpy)
    np.save("results/nfeatures_%s_%s" % (name_model, div_alpha),
            dict_ncols[div_alpha])


print("enter parallel")
backend = 'loky'
n_jobs = len(name_models) * len(div_alphas)
with parallel_backend(backend, n_jobs=n_jobs, inner_max_num_threads=1):
    Parallel()(
        delayed(parallel_function)(
            name_model, div_alpha)
        for name_model, div_alpha in product(
            name_models, div_alphas))
print('OK finished parallel')
