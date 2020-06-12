

import numpy as np

from sparse_ho.models import SVM
from sparse_ho.criterion import Logistic
from sparse_ho.forward import Forward
from sparse_ho.utils import Monitor
from sparse_ho.datasets.real import get_real_sim
# from sparse_ho.datasets.real import get_rcv1
# from sparse_ho.datasets.real import get_leukemia
from sparse_ho.grid_search import grid_search
from sparse_ho.ho import grad_search
X_train, X_val, X_test, y_train, y_val, y_test = get_real_sim(csr=True)
# X_train, X_val, X_test, y_train, y_val, y_test = get_rcv1()
# X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()
n_samples, n_features = X_train.shape

print("Starting path computation...")


n_alphas = 10
Cs = np.geomspace(1e-1, 1e4, n_alphas)
logCs = np.log(Cs)

tol = 1e-3

# model = Lasso(X_train, y_train, np.log(alpha_max/10))
# criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
# algo = Forward(criterion, use_sk=False)
# monitor_grid = Monitor()
# grid_search(
#     algo, None, None, monitor_grid, log_alphas=log_alphas,
#     tol=tol)

# grid search
model = SVM(X_train, y_train, Cs[0], max_iter=1000)
criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = Forward(criterion)
monitor_grid_sk = Monitor()
grid_search(
    algo, None, None, monitor_grid_sk, log_alphas=logCs,
    tol=tol)
monitor = Monitor()
grad_search(
        algo, logCs[0], monitor, n_outer=5, verbose=True,
        tolerance_decrease='constant', tol=1e-5,
        t_max=10000)
np.save("p_alphas.npy", Cs)
objs = np.array(monitor_grid_sk.objs)
np.save("objs.npy", objs)
