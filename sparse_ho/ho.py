# This files contains the functions to perform first order descent for HO
# hyperparameter setting

import numpy as np
from numpy.linalg import norm


def grad_search(
        algo, criterion, model, X, y, log_alpha0, monitor, n_outer=100,
        verbose=False, tolerance_decrease='constant', tol=1e-5, t_max=10000):
    """This line-search code is taken from here:
    https://github.com/fabianp/hoag/blob/master/hoag/hoag.py

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
    n_outer: int, optional (default=100).
        number of maximum updates of alpha.
    verbose: bool, optional (default=False)
        Indicates whether information about hyperparameter
        optimization process is printed or not.
    tolerance_decrease: string, optional (default="constant")
        Tolerance decrease strategy for approximate gradient.
    tol : float, optional (default=1e-5)
        Tolerance for the inner optimization solver.
    t_max: float, optional (default=10000)
        Maximum running time threshold in seconds.
    """

    def _get_val_grad(log_alpha, tol=tol, monitor=None):
        return criterion.get_val_grad(
            model, X, y, log_alpha, algo.get_beta_jac_v, tol=tol,
            monitor=monitor)

    def _proj_param(log_alpha):
        return criterion.proj_hyperparam(model, X, y, log_alpha)

    return _grad_search(
        _get_val_grad, _proj_param, log_alpha0, monitor,
        n_outer=n_outer, verbose=verbose,
        tolerance_decrease=tolerance_decrease, tol=tol, t_max=t_max)


def _grad_search(
        _get_val_grad, proj_hyperparam, log_alpha0, monitor,
        n_outer=100, verbose=False, tolerance_decrease='constant', tol=1e-5,
        t_max=10000):

    is_multiparam = isinstance(log_alpha0, np.ndarray)
    if is_multiparam:
        log_alphak = log_alpha0.copy()
        old_log_alphak = log_alphak.copy()
    else:
        log_alphak = log_alpha0
        old_log_alphak = log_alphak

    grad_norms = []

    L_log_alpha = None
    val_outer_old = np.inf

    if tolerance_decrease == 'exponential':
        seq_tol = np.geomspace(1e-2, tol, n_outer)
    else:
        seq_tol = tol * np.ones(n_outer)

    for i in range(n_outer):
        tol = seq_tol[i]
        try:
            old_tol = seq_tol[i - 1]
        except Exception:
            old_tol = seq_tol[0]
        val_outer, grad_outer = _get_val_grad(log_alphak, tol=tol,
                                              monitor=monitor)

        grad_norms.append(norm(grad_outer))
        if np.isnan(grad_norms[-1]):
            print("Nan present in gradient")
            break

        if L_log_alpha is None:
            if grad_norms[-1] > 1e-3:
                # make sure we are not selecting a step size that is too small
                if is_multiparam:
                    L_log_alpha = grad_norms[-1] / np.sqrt(len(log_alphak))
                else:
                    L_log_alpha = grad_norms[-1]
            else:
                L_log_alpha = 1
        step_size = (1. / L_log_alpha)
        try:
            old_log_alphak = log_alphak.copy()
        except Exception:
            old_log_alphak = log_alphak
        log_alphak -= step_size * grad_outer

        incr = norm(step_size * grad_outer)
        C = 0.25
        factor_L_log_alpha = 1.0
        if val_outer <= val_outer_old + C * tol + \
                old_tol * (C + factor_L_log_alpha) * incr - \
                factor_L_log_alpha * (L_log_alpha) * incr * incr:
            L_log_alpha *= 0.95
            if verbose > 1:
                print('increased step size')
            log_alphak -= step_size * grad_outer

        elif val_outer >= 1.2 * val_outer:
            if verbose > 1:
                print('decrease step size')
            # decrease step size
            L_log_alpha *= 2
            if is_multiparam:
                log_alphak = old_log_alphak.copy()
            else:
                log_alphak = old_log_alphak
            print('!!step size rejected!!', val_outer, val_outer_old)
            val_outer, grad_outer = _get_val_grad(
                log_alphak, tol=tol)

            tol *= 0.5
        else:
            old_log_alphak = log_alphak.copy()
            log_alphak -= step_size * grad_outer

        log_alphak = proj_hyperparam(log_alphak)
        val_outer_old = val_outer

        if verbose:
            print('grad outer', grad_outer)
            print('value of log_alphak', log_alphak)
        if monitor.times[-1] > t_max:
            break
    return log_alphak, val_outer, grad_outer


def grad_search_wolfe(
        algo, criterion, model, log_alpha0, monitor, n_outer=10,
        warm_start=None, tol=1e-3, maxit_ln=5):

    def _get_val_grad(log_alpha, tol=tol):
        return criterion.get_val_grad(model, log_alpha, algo.get_beta_jac_v,
                                      tol=tol)

    def _get_val(log_alpha, tol=tol):
        return criterion.get_val(model, log_alpha, tol=tol)

    log_alphak = log_alpha0
    for i in range(n_outer):
        val, grad = _get_val_grad(log_alphak)

        monitor(val.copy(), criterion.val_test, log_alphak,
                grad, criterion.rmse)

        # step_size = 1 / norm(grad)
        step_size = wolfe(
            log_alphak, -grad, val, _get_val, _get_val_grad, maxit_ln=maxit_ln)
        log_alphak -= step_size * grad


def wolfe(x_k, p_k, val, fun, fun_grad, maxit_ln=5):

    alpha_low = 0
    alpha_high = 1000
    alpha = 1 / (10 * norm(p_k))
    # alpha = 1 / (10 * norm(p_k))
    c1 = 0.1
    c2 = 0.7

    k = 0
    while k < maxit_ln:
        if (fun(x_k + alpha * p_k) > val - c1 * (alpha * norm(p_k) ** 2)):
            alpha_high = alpha
            alpha = (alpha_high+alpha_low) / 2
        elif fun_grad(x_k + alpha * p_k)[1].T * p_k < - c2 * norm(p_k) ** 2:
            alpha_low = alpha
            if alpha_high > 100:
                alpha = 2 * alpha_low
            else:
                alpha = (alpha_high + alpha_low) / 2
        else:
            break
        k += 1

    return alpha
