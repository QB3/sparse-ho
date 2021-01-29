import itertools
import numpy as np
from scipy.sparse import csc_matrix
from sklearn import linear_model
from sparse_ho.utils import Monitor
from celer.datasets import fetch_libsvm
from sparse_ho import Forward, Backward
from sparse_ho.models import Lasso, ElasticNet
from sparse_ho.tests.cvxpylayer import lasso_cvxpy, enet_cvxpy
from sparse_ho.criterion import HeldOutMSE
import time

X, y = fetch_libsvm("leukemia")
X = np.array(X.todense())
n_samples, n_features = X.shape
idx_train = np.arange(0, (3 * n_samples) //4)
idx_val = np.arange(n_samples//4, n_samples)



name_models = ["lasso"]

dict_models = {}
dict_models["lasso"] = Lasso()
dict_models["enet"] = ElasticNet()

dict_cvxpy = {}
dict_cvxpy["lasso"] = lasso_cvxpy
dict_cvxpy["enet"] = enet_cvxpy
list_nfeatures = np.geomspace(100, n_features//5, num=10, dtype=int)


tol = 1e-6
l1_ratio = 0.8
repeat = 10

# avoid compilation time
for name_model in name_models:
    index_col = np.arange(10)
    alpha_max = (np.abs(X[np.ix_(idx_train, index_col)].T @ y[idx_train])).max() / len(idx_train)
    if name_model == "lasso":
        log_alpha = np.log(alpha_max / 10)
    elif name_model == "enet":
        alpha0 = alpha_max / 10
        alpha1 = (1 - l1_ratio) * alpha0 / l1_ratio
        log_alpha = np.log(np.array([alpha0, alpha1]))


        criterion = HeldOutMSE(idx_train, idx_val)
        algo = Forward()
        monitor = Monitor()
        val, grad = criterion.get_val_grad(
        dict_models[name_model], X[:, index_col], y, log_alpha, algo.get_beta_jac_v,
        tol=tol, monitor=monitor)


        criterion = HeldOutMSE(idx_train, idx_val)
        algo = Backward()
        monitor = Monitor()
        val, grad = criterion.get_val_grad(
        dict_models[name_model], X[:, index_col], y, log_alpha, algo.get_beta_jac_v,
        tol=tol, monitor=monitor)

        val_cvxpy, grad_cvxpy = dict_cvxpy[name_model](
            X[:,index_col], y, np.exp(log_alpha), idx_train, idx_val)



for name_model in name_models:
    list_times_fwd = []
    list_times_bwd = []
    list_times_cvxpy = []
    for n_col in list_nfeatures:
        temp_fwd = []
        temp_bwd = []
        temp_cvxpy = []
        for i in range(repeat):

            rng = np.random.RandomState(i)
            index_col = rng.choice(n_features, n_col, replace=False)
            alpha_max = (np.abs(X[np.ix_(idx_train, index_col)].T @ y[idx_train])).max() / len(idx_train)
            if name_model == "lasso":
                log_alpha = np.log(alpha_max / 10)
            elif name_model == "enet":
                alpha0 = alpha_max / 10
                alpha1 = (1 - l1_ratio) * alpha0 / l1_ratio
                log_alpha = np.log(np.array([alpha0, alpha1]))


            criterion = HeldOutMSE(idx_train, idx_val)
            algo = Forward()
            monitor = Monitor()
            val, grad = criterion.get_val_grad(
                dict_models[name_model], X[:, index_col], y, log_alpha, algo.get_beta_jac_v,
                tol=tol, monitor=monitor)
            temp_fwd.append(monitor.times)


            criterion = HeldOutMSE(idx_train, idx_val)
            algo = Backward()
            monitor = Monitor()
            val, grad = criterion.get_val_grad(
                dict_models[name_model], X[:, index_col], y, log_alpha, algo.get_beta_jac_v,
                tol=tol, monitor=monitor)
            temp_bwd.append(monitor.times)

            t0 = time.time()
            val_cvxpy, grad_cvxpy = dict_cvxpy[name_model](
                X[:,index_col], y, np.exp(log_alpha), idx_train, idx_val)
            temp_cvxpy.append(time.time() - t0)
        list_times_fwd.append(np.mean(np.array(temp_fwd)))
        list_times_bwd.append(np.mean(np.array(temp_bwd)))
        list_times_cvxpy.append(np.mean(np.array(temp_cvxpy)))


    np.save("results/times_%s_forward" % name_model, list_times_fwd)
    np.save("results/times_%s_backward" % name_model, list_times_bwd)
    np.save("results/times_%s_cvxpy" % name_model, list_times_cvxpy)
    np.save("results/nfeatures_%s" % name_model, list_nfeatures)