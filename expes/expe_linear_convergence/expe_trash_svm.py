

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import norm

from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import Logistic
from sparse_ho.forward import Forward
from sparse_ho.utils import Monitor
from sparse_ho.datasets.real import get_real_sim
# from sparse_ho.datasets.real import get_rcv1
# from sparse_ho.datasets.real import get_leukemia
from sparse_ho.grid_search import grid_search
# from sparse_ho.ho import grad_search
X_train, X_val, X_test, y_train, y_val, y_test = get_real_sim(csr=False)
# X_train, X_val, X_test, y_train, y_val, y_test = get_rcv1(csr=True)

# X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()
n_samples, n_features = X_train.shape

print("Starting path computation...")

alpha_max = np.max(np.abs(X_train.T @ (- y_train)))
alpha_max /= (2 * n_samples)
n_alphas = 10
p_alphas = np.geomspace(1, 1e-4, n_alphas)
alphas = p_alphas * alpha_max
log_alphas = np.log(alphas)
tol = 1e-5

# grid search
model = SparseLogreg(X_train, y_train, log_alphas[0], max_iter=1000)
criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = Forward(criterion)
monitor_grid_sk = Monitor()
grid_search(
    algo, None, None, monitor_grid_sk, log_alphas=log_alphas,
    tol=tol)
monitor = Monitor()
# grad_search(
#     algo, logCs[0], monitor, n_outer=5, verbose=True,
#     tolerance_decrease='constant', tol=1e-8,
#     t_max=10000)

plt.figure()
plt.plot(monitor_grid_sk.log_alphas, monitor_grid_sk.objs)
plt.plot(monitor.log_alphas, monitor.objs, 'bo')
plt.show(block=False)
