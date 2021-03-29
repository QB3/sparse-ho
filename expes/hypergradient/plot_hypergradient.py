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

maxits = [5, 10, 25, 50, 75, 100]
methods = ["forward", "implicit_forward", "celer"]

dict_label = {}
dict_label["forward"] = "forward"
dict_label["implicit_forward"] = "Implicit"
dict_label["celer"] = "Implicit + celer"

dataset_name = "real-sim"

p_alpha_max = 0.1


tol = 1e-32

X, y = fetch_libsvm(dataset_name)
n_samples = len(y)

sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.3333, random_state=0)
idx_train, idx_val = sss1.split(X, y)
idx_train = idx_train[0]
idx_val = idx_val[0]

dict_res = {}


for maxit in maxits:
    for method in methods:
        print("Dataset %s, maxit %i" % (method, maxit))
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
                algo = ImplicitForward(
                    tol_jac=1e-32, n_iter_jac=maxit, use_stop_crit=False)
                algo.max_iter = maxit
                val, grad = criterion.get_val_grad(
                        model, X, y, log_alpha, algo.get_beta_jac_v, tol=1e-12,
                        monitor=monitor, max_iter=maxit)
            else:
                model = Lasso(max_iter=maxit)
                criterion = HeldOutMSE(idx_train, idx_val)
                if method == "forward":
                    algo = Forward(use_stop_crit=False)
                elif method == "implicit_forward":
                    algo = ImplicitForward(
                        tol_jac=1e-8, n_iter_jac=maxit, max_iter=maxit,
                        use_stop_crit=False)
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

        dict_res[method, maxit] = (
            dataset_name, p_alpha_max, method, maxit,
            val, grad, monitor.times[0])

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

fig_time, ax_time = plt.subplots()
fig_iter, ax_iter = plt.subplots()

for method in methods:
    grads = np.zeros(len(maxits))
    times = np.zeros(len(maxits))
    for i, maxit in enumerate(maxits):
        grads[i] = dict_res[method, maxit][5]
        print(dict_res[method, maxit][6])
        times[i] = dict_res[method, maxit][6]
    ax_time.semilogy(
        times, np.abs(grads - true_grad), label=dict_label[method])

    ax_iter.semilogy(
        maxits, np.abs(grads - true_grad), label=dict_label[method])

ax_time.set_xlabel("Time (s)")
ax_time.set_ylabel("Grad - Grad Opt")
ax_time.legend()

ax_iter.set_xlabel("Iteration")
ax_iter.set_ylabel("Grad - Grad Opt")
ax_iter.legend()

fig_iter.show()
fig_time.show()
