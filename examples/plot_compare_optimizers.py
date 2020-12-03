"""
========================
Compare outer optimizers
========================

This example shows how to perform hyperparameter optimization
for sparse logistic regression using a held-out test set.

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)


from libsvmdata.datasets import fetch_libsvm
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import make_classification
from celer import LogisticRegression

from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import HeldOutLogistic
from sparse_ho import ImplicitForward
from sparse_ho import Forward
from sparse_ho.grid_search import grid_search
from sparse_ho.optimizers import LineSearch, GradientDescent, Adam


print(__doc__)

dataset = 'rcv1_train'
# dataset = 'simu'

if dataset != 'simu':
    X, y = fetch_libsvm(dataset)
    X = X[:, :100]
else:
    X, y = make_classification(
        n_samples=100, n_features=1_000, random_state=42, flip_y=0.02)


n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)

print("Starting path computation...")
n_samples = len(y[idx_train])
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train])))

alpha_max /= 4 * len(idx_train)
log_alpha_max = np.log(alpha_max)
log_alpha_min = np.log(alpha_max / 100)
max_iter = 100

log_alpha0 = np.log(0.1 * alpha_max)
tol = 1e-8

n_alphas = 30
p_alphas = np.geomspace(1, 0.0001, n_alphas)
alphas = alpha_max * p_alphas
log_alphas = np.log(alphas)

##############################################################################
# Grid-search
# -----------

estimator = LogisticRegression(
    penalty='l1', fit_intercept=False, max_iter=max_iter)
model = SparseLogreg(max_iter=max_iter, estimator=estimator)
criterion = HeldOutLogistic(idx_train, idx_val)
algo_grid = Forward()
monitor_grid = Monitor()
grid_search(
    algo_grid, criterion, model, X, y, log_alpha_min, log_alpha_max,
    monitor_grid, log_alphas=log_alphas, tol=tol)
objs = np.array(monitor_grid.objs)


##############################################################################
# Grad-search
# -----------
optimizer_names = ['line-search', 'gradient-descent', 'adam']
optimizers = {
    'line-search': LineSearch(n_outer=10, tol=tol),
    'gradient-descent': GradientDescent(n_outer=10, step_size=100),
    'adam': Adam(n_outer=10, lr=0.11, verbose=True)}

monitors = {}

for optimizer_name in optimizer_names:
    estimator = LogisticRegression(
        penalty='l1', fit_intercept=False, solver='saga', tol=tol)
    model = SparseLogreg(max_iter=max_iter, estimator=estimator)
    criterion = HeldOutLogistic(idx_train, idx_val)

    monitor_grad = Monitor()
    algo = ImplicitForward(tol_jac=tol, n_iter_jac=1000)

    optimizer = optimizers[optimizer_name]
    grad_search(
        algo, criterion, model, optimizer, X, y, log_alpha0,
        monitor_grad)
    monitors[optimizer_name] = monitor_grad


current_palette = sns.color_palette("colorblind")
dict_colors = {
    'line-search': current_palette[2],
    'gradient-descent': current_palette[3],
    'adam': current_palette[4]}

plt.figure(figsize=(5, 3))
plt.semilogx(
    p_alphas, objs, color=current_palette[0])
plt.semilogx(
    p_alphas, objs, 'bo', label='0-order method (grid-search)',
    color=current_palette[1])
for optimizer_name in optimizer_names:
    monitor = monitors[optimizer_name]
    p_alphas_grad = np.exp(np.array(monitor.log_alphas)) / alpha_max
    objs_grad = np.array(monitor.objs)
    plt.semilogx(
        p_alphas_grad, objs_grad, 'bX', label=optimizer_name,
        color=dict_colors[optimizer_name], markersize=7)
    plt.xlabel(r"$\lambda / \lambda_{\max}$")
    plt.ylabel(
        r"$ \sum_i^n \log \left ( 1 + e^{-y_i^{\rm{val}} X_i^{\rm{val}} "
        r"\hat \beta^{(\lambda)} } \right ) $")

plt.tick_params(width=5)
plt.legend(loc=1)
plt.tight_layout()
plt.show(block=False)
