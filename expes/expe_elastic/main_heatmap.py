from sparse_ho.datasets.real import get_data
import numpy as np
from sklearn.linear_model import ElasticNet as ElasticNet_sk
from numpy.linalg import norm
import matplotlib.pyplot as plt
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.criterion import CV
from sparse_ho.models import ElasticNet
from sparse_ho.ho import grad_search
from bcdsugar.utils import Monitor
from mpl_toolkits import mplot3d


dataset_name = "rcv1"
X_train, X_val, X_test, y_train, y_val, y_test = get_data(dataset_name)

alpha_max = np.max(np.abs(X_train.T @ y_train))
alpha_max /= X_train.shape[0]
log_alpha_max = np.log(alpha_max)

alpha_min = 1e-4 * alpha_max

n_grid = 10
alphas_1 = np.geomspace(alpha_min, 0.6 * alpha_max, n_grid)
log_alphas_1 = np.log(alphas_1)
alphas_2 = np.geomspace(alpha_min, 0.6 * alpha_max, n_grid)
log_alphas_2 = np.log(alphas_2)

results = np.zeros((n_grid, n_grid))
tol = 1e-5
max_iter = 50000
n_outer = 10
monitor = Monitor()

# grid search with scikit
for i in range(n_grid):
    for j in range(n_grid):
        clf = ElasticNet_sk(
            alpha=(alphas_1[i] + alphas_2[j]), fit_intercept=False,
            l1_ratio=alphas_1[i] / (alphas_1[i] + alphas_2[j]),
            tol=1e-7, max_iter=max_iter, warm_start=True)
        clf.fit(X_train, y_train)
        results[i, j] = norm(y_val - X_val @ clf.coef_) ** 2 / X_val.shape[0]

# grad search
model = ElasticNet(
    X_train, y_train, log_alphas_1[-1], log_alphas_2[-1], log_alpha_max, max_iter=max_iter, tol=tol)
criterion = CV(
    X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = ImplicitForward(
    criterion, tol_jac=1e-5, n_iter_jac=1000, use_sk=True,
    max_iter=max_iter)
_, _, _ = grad_search(
    algo=algo, verbose=False,
    log_alpha0=np.array([log_alphas_1[8], log_alphas_2[8]]), tol=tol,
    n_outer=n_outer, monitor=monitor,
    tolerance_decrease='constant')
alphas_grad = np.exp(np.array(monitor.log_alphas))
alphas_grad /= alpha_max


X, Y = np.meshgrid(alphas_1 / alpha_max, alphas_2 / alpha_max)
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(
    np.log(X), np.log(Y), results, rstride=1, cstride=1,
    cmap='viridis', edgecolor='none', alpha=0.5)
ax.scatter3D(
    np.log(alphas_grad[:, 0]), np.log(alphas_grad[:, 1]),
    monitor.objs, c="red", s=200, marker="X")
ax.set_xlabel("lambda1")
ax.set_ylabel("lambda2")
ax.set_zlabel("Loss on validation set")
fig.show()
