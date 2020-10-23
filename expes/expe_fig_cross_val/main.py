import numpy as np
from sklearn import linear_model

from sparse_ho.models import Lasso
from sparse_ho.criterion import CV
# from sparse_ho.forward import Forward
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search
from sparse_ho.datasets.real import get_real_sim
# from sparse_ho.datasets.real import get_leukemia
# from sparse_ho.grid_search import grid_search

X_train, X_val, X_test, y_train, y_val, y_test = get_real_sim()
# X_train, X_val, X_test, y_train, y_val, y_test = get_leukemia()
n_samples, n_features = X_train.shape

print("Starting path computation...")
n_samples = len(y_train)
alpha_max = np.max(np.abs(X_train.T.dot(y_train))) / X_train.shape[0]
log_alpha0 = np.log(alpha_max / 10)

n_alphas = 10
p_alphas = np.geomspace(1, 0.001, n_alphas)
alphas = alpha_max * p_alphas
log_alphas = np.log(alphas)

tol = 1e-7

# grid search
# model = Lasso(X_train, y_train, np.log(alpha_max/10))
# criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
# algo = Forward(criterion)
# monitor_grid_sk = Monitor()
# grid_search(
#     algo, None, None, monitor_grid_sk, log_alphas=log_alphas,
#     tol=tol)

# np.save("p_alphas.npy", p_alphas)
# objs = np.array(monitor_grid_sk.objs)
# np.save("objs.npy", objs)

# grad_search
estimator = linear_model.Lasso(
    fit_intercept=False, warm_start=True)
model = Lasso(X_train, y_train, np.log(alpha_max/10), estimator=estimator)
criterion = CV(X_val, y_val, model, X_test=X_test, y_test=y_test)
algo = ImplicitForward(criterion)
monitor_grad = Monitor()
grad_search(
    algo, np.log(alpha_max/10), monitor_grad, n_outer=10, tol=tol)


p_alphas_grad = np.exp(np.array(monitor_grad.log_alphas)) / alpha_max

np.save("p_alphas_grad.npy", p_alphas_grad)
objs_grad = np.array(monitor_grad.objs)
np.save("objs_grad.npy", objs_grad)
