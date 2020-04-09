import numpy as np
from sparse_ho.ho import get_val_grad
from sparse_ho.utils import Monitor, WarmStart
from hyperopt import hp
from hyperopt import fmin, tpe, rand


def hyperopt_lasso(
        X_train, y_train, log_alpha, X_val, y_val, X_test, y_test, tol,
        maxit=1000, max_evals=30, method="bayesian", criterion="cv", sigma=1.0,
        beta_star=None):
    n_samples, n_features = X_train.shape
    alpha_max = np.abs((X_train.T @ y_train)).max() / n_samples

    space = hp.uniform(
        'log_alpha', np.log(alpha_max / 1000), np.log(alpha_max))

    monitor = Monitor()
    warm_start = WarmStart()

    if criterion == "cv":
        def objective(log_alpha):
            value = get_val_grad(
                X_train, y_train, log_alpha, X_val, y_val, X_test, y_test, tol,
                monitor, warm_start, method="hyperopt", maxit=1000,
                model="lasso", beta_star=beta_star)
            return value
    elif criterion == "sure":
        def objective(log_alpha):
            value = get_val_grad(
                X_train, y_train, log_alpha, X_val, y_val, X_test, y_test, tol,
                monitor, warm_start, method="hyperopt", maxit=1000,
                model="lasso", criterion="sure", sigma=sigma,
                beta_star=beta_star)
            return value

    if method == "bayesian":
        fmin(objective, space, algo=tpe.suggest, max_evals=max_evals)
    elif method == "random":
        fmin(objective, space, algo=rand.suggest, max_evals=max_evals)
    return monitor
