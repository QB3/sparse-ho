# This files contains the functions to perform first order descent for HO
# hyperparameter setting

import numpy as np
from numpy.linalg import norm


def grad_search(
        algo, log_alpha0, monitor, n_outer=100, verbose=False,
        tolerance_decrease='constant', tol=1e-5,
        convexify=False, gamma_convex=False, beta_star=None, t_max=10000):
    """This line-search code is taken from here:
    https://github.com/fabianp/hoag/blob/master/hoag/hoag.py

    Parameters
    ----------
    log_alpha0: float
        log of the regularization coefficient alpha
    tol : float
        tolerance for the inner optimization solver
    monitor: Monitor object
        used to store the value of the cross-validation function
    n_outer: int
        number of maximum iteration in the outer loop (for the line search)
    tolerance_decrease: string
        tolerance decrease strategy for approximate gradient
    convexify: bool
        True if you want to regularize the problem
    gamma: non negative float
        convexification coefficient
    criterion: string
        criterion to optimize during hyperparameter optimization
        you may choose between "cv" and "sure"
    gamma_sure:
        constant for sure problem
     sigma,
        constant for sure problem
    random_state: int
    beta_star: np.array, shape (n_features,)
        True coefficients of the underlying model (if known)
        used to compute metrics
    """

    def _get_val_grad(lambdak, tol=tol):
        return algo.get_val_grad(lambdak, tol=tol, beta_star=beta_star)

    def _proj_param(lambdak):
        return algo.criterion.model.proj_param(lambdak)

    return _grad_search(
        _get_val_grad, _proj_param, log_alpha0, monitor, algo, n_outer=n_outer,
        verbose=verbose, tolerance_decrease=tolerance_decrease, tol=tol,
        t_max=t_max)


def _grad_search(
        _get_val_grad, proj_param, log_alpha0, monitor, algo, n_outer=100,
        verbose=False, tolerance_decrease='constant', tol=1e-5,
        convexify=False, gamma_convex=False, beta_star=None, t_max=10000):
    """
    This line-search code is taken from here:
    https://github.com/fabianp/hoag/blob/master/hoag/hoag.py

    Parameters
    --------------
    X_train: np.array, shape (n_samples, n_features)
        observation used for training

    y_train: np.array, shape (n_samples, n_features)
        targets used for training

    log_alpha: float
        log of the regularization coefficient alpha

    X_val: np.array, shape (n_samples, n_features)
        observation used for cross-validation

    y_val: np.array, shape (n_samples, n_features)
        targets used for cross-validation

    X_test: np.array, shape (n_samples, n_features)
        observation used for testing

    y_test: np.array, shape (n_samples, n_features)
        targets used for testing

    tol : float
        tolerance for the inner optimization solver
    monitor: Monitor object
        used to store the value of the cross-validation function

    warm_start: WarmStart object
        used for warm start for all methods

    method: string
        method used to compute the hypergradient, you may want to use

        "implicit" "forward" "backward" "fast_forward_iterdiff"
    maxit: int
        maximum number of iterations in the inner optimization solver

    n_outer: int
        number of maximum iteration in the outer loop (for the line search)

    tolerance_decrease: string
        tolerance decrease strategy for approximate gradient

    niter_jac: int
        maximum number of iteration for the fast_forward_iterdiff
        method in the Jacobian computation

    model: string
        model used, "lasso", "wlasso", "mcp"

    tol_jac: float
        tolerance for the Jacobian loop

    convexify: bool
        True if you want to regularize the problem

    gamma: non negative float
        convexification coefficient

    criterion: string
        criterion to optimize during hyperparameter optimization
        you may choose between "cv" and "sure"

    C: float
        constant for sure problem

    gamma_sure:
        constant for sure problem

     sigma,
        constant for sure problem

    random_state: int

    beta_star: np.array, shape (n_features,)
        True coefficients of the underlying model (if known)
        used to compute metrics
    """

    try:
        lambdak = log_alpha0.copy()
        old_lambdak = lambdak.copy()
    except Exception:
        lambdak = log_alpha0
        old_lambdak = lambdak
    old_grads = []

    L_lambda = None
    g_func_old = np.inf

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
        g_func, grad_lambda = _get_val_grad(lambdak, tol=tol)
        try:
            monitor(g_func, algo.criterion.val_test, lambdak.copy(),
                    grad_lambda, algo.criterion.rmse)
        except Exception:
            monitor(g_func, algo.criterion.val_test, lambdak,
                    grad_lambda, algo.criterion.rmse)

        # TODO this should be removed into the SURE class no?
        if convexify:
            g_func += gamma_convex * np.sum(np.exp(lambdak) ** 2)
            grad_lambda += gamma_convex * np.exp(lambdak)

        old_grads.append(norm(grad_lambda))
        if np.isnan(old_grads[-1]):
            print("Nan present in gradient")
            break

        if L_lambda is None:
            if old_grads[-1] > 1e-3:
                # make sure we are not selecting a step size that is too small
                try:
                    L_lambda = old_grads[-1] / np.sqrt(len(lambdak))
                except Exception:
                    L_lambda = old_grads[-1]
            else:
                L_lambda = 1
        step_size = (1. / L_lambda)
        try:
            old_lambdak = lambdak.copy()
        except Exception:
            old_lambdak = lambdak
        lambdak -= step_size * grad_lambda

        incr = norm(step_size * grad_lambda)
        C = 0.25
        # C = 0.25 / algo.criterion.model.X.shape[0]
        factor_L_lambda = 1.0
        if g_func <= g_func_old + C * tol + \
                old_tol * (C + factor_L_lambda) * incr - factor_L_lambda * \
                (L_lambda) * incr * incr:
            L_lambda *= 0.95
            if verbose > 1:
                print('increased step size')
            lambdak -= step_size * grad_lambda

        elif g_func >= 1.2 * g_func_old:
            if verbose > 1:
                print('decrease step size')
            # decrease step size
            L_lambda *= 2
            try:
                lambdak = old_lambdak.copy()
            except Exception:
                lambdak = old_lambdak
            print('!!step size rejected!!', g_func, g_func_old)
            g_func, grad_lambda = _get_val_grad(
                lambdak, tol=tol)
            # g_func, grad_lambda = algo.get_val_grad(lambdak, tol=tol,
            #
            #                                         beta_star=beta_star)
            if convexify:
                g_func += gamma_convex * np.sum(np.exp(lambdak) ** 2)
                grad_lambda += gamma_convex * np.exp(lambdak)
            tol *= 0.5
        else:
            old_lambdak = lambdak.copy()
            lambdak -= step_size * grad_lambda

        lambdak = proj_param(lambdak)

        g_func_old = g_func

        if verbose:
            print('grad lambda', grad_lambda)
            print('value of lambda_k', lambdak)
        if monitor.times[-1] > t_max:
            break
    return lambdak, g_func, grad_lambda


def grad_search_wolfe(
        algo, log_alpha0, monitor, n_outer=10, warm_start=None, tol=1e-3,
        maxit_ln=5):

    def _get_val_grad(lambdak, tol=tol):
        return algo.get_val_grad(lambdak, tol=tol)

    def _get_val(lambdak, tol=tol):
        return algo.criterion.get_val(lambdak, tol=tol)

    log_alphak = log_alpha0
    for i in range(n_outer):
        val, grad = _get_val_grad(log_alphak)

        monitor(val.copy(), algo.criterion.val_test, log_alphak,
                grad, algo.criterion.rmse)

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
