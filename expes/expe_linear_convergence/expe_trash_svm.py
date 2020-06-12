

import numpy as np
import matplotlib.pyplot as plt

from scipy.sparse.linalg import norm

from sparse_ho.models import SVM
from sparse_ho.criterion import Logistic
from sparse_ho.forward import Forward
from sparse_ho.utils import Monitor
# from sparse_ho.datasets.real import get_real_sim
from sparse_ho.datasets.real import get_rcv1
# from sparse_ho.datasets.real import get_leukemia
from sparse_ho.grid_search import grid_search
# from sparse_ho.ho import grad_search
# X_train, X_val, X_test, y_train, y_val, y_test = get_real_sim(csr=True)
X_train, X_val, X_test, y_train, y_val, y_test = get_rcv1(csr=True)

X_train /= norm(X_train, axis=0)
X_val /= norm(X_val, axis=0)
X_test /= norm(X_test, axis=0)

# X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()
n_samples, n_features = X_train.shape

print("Starting path computation...")


n_alphas = 10
Cs = np.geomspace(1e-3, 1e-1, n_alphas)
logCs = np.log(Cs)

tol = 1e-3

# grid search
model = SVM(X_train, y_train, Cs[0], max_iter=1000)
criterion = Logistic(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = Forward(criterion)
monitor_grid_sk = Monitor()
grid_search(
    algo, None, None, monitor_grid_sk, log_alphas=logCs,
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
