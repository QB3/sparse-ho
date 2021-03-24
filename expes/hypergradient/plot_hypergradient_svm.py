import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from lightning.classification import LinearSVC
from libsvmdata import fetch_libsvm
from scipy.sparse.linalg import norm
from celer.datasets import make_correlated_data

from sparse_ho.models import SVM
from sparse_ho.criterion import HeldOutSmoothedHinge
from sparse_ho import ImplicitForward, Implicit
from sparse_ho import Forward, Backward
from sparse_ho.utils import Monitor

maxits = [5, 10, 25, 50, 75, 100, 500, 1000]
methods = ["forward", "implicit_forward", "sota"]

dict_label = {}
dict_label["forward"] = "forward"
dict_label["implicit_forward"] = "Implicit"
dict_label["sota"] = "Implicit + sota"

dataset_name = "real-sim"

logC = np.log(0.1)

tol = 1e-32

X, y, w_true = make_correlated_data(
    n_samples=3_000, n_features=100, random_state=0, snr=5)
y = np.sign(y)
# print(X.sum())

# X, y = fetch_libsvm(dataset_name)
# X = X[:, :10]
# X = csr_matrix(X)  # very important for SVM
# my_bool = norm(X, axis=1) != 0
# X = X[my_bool, :]
# y = y[my_bool]

sss1 = StratifiedShuffleSplit(n_splits=2, test_size=0.3333, random_state=0)
idx_train, idx_val = sss1.split(X, y)
idx_train = idx_train[0]
idx_val = idx_val[0]


true_monitor = Monitor()
clf = LinearSVC(
        C=np.exp(logC), tol=1e-32, max_iter=10_000, loss='hinge',
        permute=False)
criterion = HeldOutSmoothedHinge(idx_train, idx_val)
algo = Implicit(criterion)
model = SVM(estimator=clf, max_iter=10_000)
true_val, true_grad = criterion.get_val_grad(
        model, X, y, logC, algo.get_beta_jac_v, tol=1e-14,
        monitor=true_monitor)

dict_res = {}
for max_iter in maxits:
    for method in methods:
        print("Dataset %s, maxit %i" % (method, max_iter))
        for i in range(2):
            monitor = Monitor()
            model = SVM(max_iter=max_iter)
            criterion = HeldOutSmoothedHinge(idx_train, idx_val)
            if method == "sota":
                clf = LinearSVC(
                    C=np.exp(logC), loss='hinge', max_iter=max_iter, tol=1e-32,
                    permute=False)
                model.estimator = clf
                algo = ImplicitForward(
                    tol_jac=1e-32, n_iter_jac=max_iter, use_stop_crit=False)
                algo.max_iter = max_iter
                val, grad = criterion.get_val_grad(
                        model, X, y, logC, algo.get_beta_jac_v, tol=1e-12,
                        monitor=monitor, max_iter=max_iter)
            else:
                if method == "forward":
                    algo = Forward(use_stop_crit=False)
                elif method == "implicit_forward":
                    algo = ImplicitForward(
                        tol_jac=1e-8, n_iter_jac=max_iter, max_iter=max_iter,
                        use_stop_crit=False)
                elif method == "implicit":
                    algo = Implicit(max_iter=1000)
                elif method == "backward":
                    algo = Backward()
                else:
                    raise NotImplementedError
                algo.max_iter = max_iter
                algo.use_stop_crit = False
                val, grad = criterion.get_val_grad(
                        model, X, y, logC, algo.get_beta_jac_v, tol=tol,
                        monitor=monitor, max_iter=max_iter)

        dict_res[method, max_iter] = (
            dataset_name, logC, method, max_iter,
            val, grad, monitor.times[0])

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
