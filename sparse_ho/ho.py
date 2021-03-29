
# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#
# License: BSD (3-clause)

# This files contains the functions to perform first order descent for HO
# hyperparameter setting


import numpy as np
from hyperopt import hp
from hyperopt import fmin, tpe, rand
from functools import partial
from sklearn.utils import check_random_state


def grad_search(
        algo, criterion, model, optimizer, X, y, alpha0, monitor):
    """
    Parameters
    ----------
    algo: instance of BaseAlgo
        algorithm used to compute hypergradient.
    criterion:  instance of BaseCriterion
        criterion to optimize during hyperparameter optimization
        (outer optimization problem).
    model: instance of BaseModel
        model on which hyperparameter has to be selected
        (inner optimization problem).
    optimizer: instance of Optimizer
        optimizer used to minimize the criterion (outer optimization)
    X: array like of shape (n_samples, n_features)
        Design matrix.
    y: array like of shape (n_samples,)
        Target.
    alpha0: float
        initial value of the hyperparameter alpha.
    monitor: instance of Monitor
        used to store the value of the cross-validation function.


    Returns
    -------
    XXX missing
    """

    def _get_val_grad(log_alpha, tol, monitor):
        return criterion.get_val_grad(
            model, X, y, log_alpha, algo.compute_beta_grad, tol=tol,
            monitor=monitor)

    def _proj_hyperparam(log_alpha):
        return criterion.proj_hyperparam(model, X, y, log_alpha)

    return optimizer._grad_search(
        _get_val_grad, _proj_hyperparam, np.log(alpha0), monitor)


def hyperopt_wrapper(
        algo, criterion, model, X, y, alpha_min, alpha_max, monitor,
        max_evals=50, tol=1e-5, random_state=42, t_max=100_000,
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
    alpha_min: float
        minimum value for the regularization coefficient alpha.
    alpha_max: float
        maximum value for the regularization coefficient alpha.
    monitor: instance of Monitor
        used to store the value of the cross-validation function.
    max_evals: int (default=50)
        maximum number of evaluation of the function
    tol: float (default=1e-5)
        tolerance for TODO
    random_state: int or instance of RandomState
        Random number generator used for reproducibility.
    t_max: int, optional (default=100_000)
        TODO
    method: 'random' | 'bayesian' (default='bayesian')
        method for hyperopt
    size_space: int (default=1)
        size of the hyperparameter space

    Returns
    -------
    monitor:
        The instance of Monitor used during iterations.
    """

    def objective(log_alpha):
        log_alpha = np.array(log_alpha)
        val_func = criterion.get_val(
            model, X, y, log_alpha, monitor, tol=tol)
        return val_func

    # TODO, also size_space = n_hyperparam ?
    space = [
        hp.uniform(str(dim), np.log(alpha_min), np.log(alpha_max)) for
        dim in range(size_space)]

    rng = check_random_state(random_state)

    if method == "bayesian":
        algo = partial(tpe.suggest, n_startup_jobs=5)
        fmin(
            objective, space, algo=algo, max_evals=max_evals,
            timeout=t_max, rstate=rng)
    elif method == "random":
        fmin(
            objective, space, algo=rand.suggest, max_evals=max_evals,
            timeout=t_max, rstate=rng)
    return monitor
