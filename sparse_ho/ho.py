
# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

# This files contains the functions to perform first order descent for HO
# hyperparameter setting


import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, rand
from sklearn.utils import check_random_state


def grad_search(
        algo, criterion, model, optimizer, X, y, log_alpha0, monitor):
    """
    Parameters
    ----------
    algo: instance of BaseAlgo
        algorithm used to compute hypergradient.
    criterion:  instance of BaseCriterion
        criterion to optimize during hyperparameter optimization
        (outer optimization problem).
    model:  instance of BaseModel
        model on which hyperparameter has to be selected
        (inner optimization problem).
    X: array like of shape (n_samples, n_features)
        Design matrix.
    y: array like of shape (n_samples,)
        Target.
    log_alpha0: float
        initial value of the logarithm of the regularization coefficient alpha.
    monitor: instance of Monitor
        used to store the value of the cross-validation function.
    """

    def _get_val_grad(log_alpha, tol, monitor):
        return criterion.get_val_grad(
            model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol,
            monitor=monitor)

    def _proj_hyperparam(log_alpha):
        return criterion.proj_hyperparam(model, X, y, log_alpha)

    return optimizer._grad_search(
        _get_val_grad, _proj_hyperparam, log_alpha0, monitor)


def hyperopt_wrapper(
        algo, criterion, model, X, y, log_alpha_min, log_alpha_max, monitor,
        max_evals=50, tol=1e-5, random_state=42, t_max=1000,
        method='bayesian', size_space=1):
    """
    Parameters
    ----------
    algo: instance of BaseAlgo
        algorithm used to compute hypergradient.
    criterion:  instance of BaseCriterion
        criterion to optimize during hyperparameter optimization
        (outer optimization problem).
    model:  instance of BaseModel
        model on which hyperparameter has to be selected
        (inner optimization problem).
    X: array like of shape (n_samples, n_features)
        Design matrix.
    y: array like of shape (n_samples,)
        Target.
    log_alpha_min: float
        minimum value for the regularization coefficient alpha.
    log_alpha_max: float
        maximum value for the regularization coefficient alpha.
    monitor: instance of Monitor
        used to store the value of the cross-validation function.
    max_evals: int (default=50)
        maximum number of evaluation of the function
    tol: float (default=1e-5)
    random_state=42
    t_max=1000,
    method=string (default='bayesian')
        method for hyperopt, 'random' or 'bayesian'
    size_space: int (default=1)
        size of the hyperparameter space
    """

    def objective(log_alpha):
        log_alpha = np.array(log_alpha)
        val_func = criterion.get_val(
            model, X, y, log_alpha, monitor, tol=tol)
        return val_func

    space = [
        hp.uniform(str(dim), log_alpha_min, log_alpha_max) for dim in range(
            size_space)]

    # space = hp.uniform(
    #     'log_alpha', log_alpha_min, log_alpha_max)

    rng = check_random_state(random_state)

    if method == "bayesian":
        fmin(
            objective, space, algo=tpe.suggest, max_evals=max_evals,
            timeout=t_max, rstate=rng)
    elif method == "random":
        fmin(
            objective, space, algo=rand.suggest, max_evals=max_evals,
            timeout=t_max, rstate=rng)
    return monitor
