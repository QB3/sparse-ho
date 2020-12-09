import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit

from celer import Lasso as Lasso_celer
from libsvmdata import fetch_libsvm

from sparse_ho.models import Lasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import ImplicitForward, Implicit
from sparse_ho import Forward, Backward
from sparse_ho.utils import Monitor

maxits = [1, 2, 3, 4, 5, 10]
methods = ["forward", "implicit_forward", "celer"]
dataset_name = "real-sim"

p_alpha_max = 0.1


tol = 1e-16
dict_algo = {}
dict_algo["forward"] = Forward()
dict_algo["implicit_forward"] = ImplicitForward(
    tol_jac=1e-8, n_iter_jac=100000, max_iter=1000)
dict_algo["implicit"] = Implicit(max_iter=1000)
dict_algo["backward"] = Backward()

X, y = fetch_libsvm(dataset_name)
n_samples = len(y)

sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.3333, random_state=0)
idx_train, idx_val = sss1.split(X, y)
idx_train = idx_train[0]
idx_val = idx_val[0]

dict_res = {}



for maxit in maxits:
    for method in methods:
        for i in range(2):
            alpha_max = np.max(np.abs(X.T.dot(y))) / n_samples
            log_alpha = np.log(alpha_max * p_alpha_max)
            monitor = Monitor()
            if method == "celer":
                clf = Lasso_celer(
                    alpha=np.exp(log_alpha), fit_intercept=False,
                    tol=1e-12, max_iter=maxit)
                model = Lasso(estimator=clf, max_iter=maxit)
                criterion = HeldOutMSE(idx_train, idx_val)
                algo = ImplicitForward(tol_jac=1e-8, n_iter_jac=maxit)
                algo.max_iter = maxit
                val, grad = criterion.get_val_grad(
                        model, X, y, log_alpha, algo.get_beta_jac_v, tol=1e-12,
                        monitor=monitor, max_iter=maxit)
            else:
                model = Lasso(max_iter=maxit)
                criterion = HeldOutMSE(idx_train, idx_val)
                algo = dict_algo[method]
                algo.max_iter = maxit
                val, grad = criterion.get_val_grad(
                        model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol,
                        monitor=monitor, max_iter=maxit)

            true_monitor = Monitor()
            clf = Lasso_celer(
                    alpha=np.exp(log_alpha), fit_intercept=False,
                    warm_start=True, tol=1e-14, max_iter=10000)
            criterion = HeldOutMSE(idx_train, idx_val)
            algo = Implicit(criterion)
            model = Lasso(estimator=clf, max_iter=10000)
            true_val, true_grad = criterion.get_val_grad(
                    model, X, y, log_alpha, algo.get_beta_jac_v, tol=1e-14,
                    monitor=true_monitor)

        dict_res[method, maxit] = (
            dataset_name, p_alpha_max, method, maxit,
            val, grad, monitor.times[0], true_val, true_grad,
            true_monitor.times[0])


plt.figure()
for method in methods:
    grads = np.zeros(len(maxits))
    times = np.zeros(len(maxits))
    for i, maxit in enumerate(maxits):
        grads[i] = dict_res[method, maxit][6]
        times[i] = dict_res[method, maxit][7]
    plt.plot(times, grads)

plt.show(block=False)
