"""
=======================================================================
Lasso on the held-out loss
=======================================================================
...

"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sparse_ho.models import Lasso
from sparse_ho.criterion import CV
from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search
from sparse_ho.grid_search import grid_search
# from sparse_ho.datasets.real import get_real_sim
from sparse_ho.datasets.real import get_rcv1


X_train, X_val, X_test, y_train, y_val, y_test = get_rcv1()
# X_train, X_val, X_test, y_train, y_val, y_test = get_real_sim()
n_samples, n_features = X_train.shape

print("Starting path computation...")
n_samples = len(y_train)
alpha_max = np.max(np.abs(X_train.T.dot(y_train))) / X_train.shape[0]
log_alpha0 = np.log(alpha_max / 10)

n_alphas = 10
p_alphas = np.geomspace(1, 0.0001, n_alphas)
alphas = alpha_max * p_alphas
log_alphas = np.log(alphas)

tol = 1e-7

##############################################################################
# Grid-search
# -----------
model = Lasso(X_train, y_train, np.log(alpha_max / 10))
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = Forward(criterion, use_sk=True)
monitor_grid_sk = Monitor()
grid_search(
    algo, None, None, monitor_grid_sk, log_alphas=log_alphas,
    tol=tol)
objs = np.array(monitor_grid_sk.objs)

##############################################################################
# Grad-search
# -----------
model = Lasso(X_train, y_train, np.log(alpha_max / 10))
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = ImplicitForward(criterion, use_sk=True)
monitor_grad = Monitor()
grad_search(
    algo, np.log(alpha_max / 10), monitor_grad, n_outer=10, tol=tol)


##############################################################################
# Plot results
# ------------
p_alphas_grad = np.exp(np.array(monitor_grad.log_alphas)) / alpha_max

objs_grad = np.array(monitor_grad.objs)

current_palette = sns.color_palette("colorblind")

fig = plt.figure(figsize=(5, 3))
plt.semilogx(
    p_alphas, objs, color=current_palette[0])
plt.semilogx(
    p_alphas, objs, 'bo', label='0-order method (grid-search)',
    color=current_palette[1])
plt.semilogx(
    p_alphas_grad, objs_grad, 'bX', label='1-st order method',
    color=current_palette[2])
plt.xlabel(r"$\lambda / \lambda_{\max}$")
plt.ylabel(
    r"$\|y^{\rm{val}} - X^{\rm{val}} \hat \beta^{(\lambda)} \|^2$")
plt.tick_params(width=5)
plt.legend(loc=1)
plt.tight_layout()
plt.show(block=False)
