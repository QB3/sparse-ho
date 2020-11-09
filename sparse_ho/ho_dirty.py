
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fmin_l_bfgs_b


def lbfgs(
        criterion, log_alpha0, monitor, n_outer=10, verbose=False,
        tolerance_decrease='constant', tol=1e-5,
        beta_star=None, t_max=10000):
    """
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
    TODO: convexify and gamma should be remove no? beta_star also?
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
        return criterion.get_val_grad(lambdak, tol=tol)

    return fmin_l_bfgs_b(
        _get_val_grad, log_alpha0, fprime=None, maxiter=n_outer)


def grad_search(
        criterion, log_alpha0, monitor, n_outer=10, verbose=False,
        tolerance_decrease='constant', tol=1e-5,
        beta_star=None, t_max=10000):
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
    TODO: convexify and gamma should be remove no? beta_star also?
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
        return criterion.get_val_grad(lambdak, tol=tol, monitor=monitor)

    def _proj_param(lambdak):
        return criterion.proj_param(lambdak)

    return _grad_search(
        _get_val_grad, _proj_param, log_alpha0, monitor, n_outer=n_outer,
        verbose=verbose, tolerance_decrease=tolerance_decrease, tol=tol,
        t_max=t_max)


def _grad_search(
        _get_val_grad, proj_param, log_alpha0, monitor, n_outer=100,
        verbose=False, tolerance_decrease='constant', tol=1e-5,
        beta_star=None, t_max=10000):
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

        print("%i / %i  || crosss entropy %f  || accuracy %f" % (
              i, n_outer, g_func, monitor.acc_vals[-1]))
        if verbose >= 1:
            print("outer function value %f" % g_func)
        try:
            # as in scipy I think we should use callback function, instead of rmse attriburtes, wdyt?
            monitor(g_func, None, lambdak.copy(),
                    grad_lambda)
        except Exception:
            monitor(g_func, None, lambdak, grad_lambda)

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

            tol *= 0.5
        else:
            old_lambdak = lambdak.copy()
            lambdak -= step_size * grad_lambda

        lambdak = proj_param(lambdak)
        g_func_old = g_func

        if verbose > 1:
            print('grad lambda', grad_lambda)
            print('value of lambda_k', lambdak)
        if monitor.times[-1] > t_max:
            break
    return lambdak, g_func, grad_lambda


def adam_search(
        criterion, log_alpha0, monitor, n_outer=10, verbose=False,
        tolerance_decrease='constant', tol=1e-5,
        beta_star=None, t_max=10000, epsilon=1e-3, lr=0.01, beta_2=0.999):
    """This adam code is taken from here:
    https://github.com/sagarvegad/Adam-optimizer/blob/master/Adam.py

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
        return criterion.get_val_grad(lambdak, tol=tol, monitor=monitor)

    # def _proj_param(lambdak):
    #     return criterion.proj_param(lambdak)

    return _adam_search(
        _get_val_grad, log_alpha0, monitor, n_outer=n_outer,
        verbose=verbose, tolerance_decrease=tolerance_decrease, tol=tol,
        t_max=t_max, epsilon=epsilon, lr=lr, beta_2=beta_2)


def _adam_search(
        _get_val_grad, log_alpha0, monitor, n_outer=100,
        verbose=False, tolerance_decrease='constant', tol=1e-5,
        beta_star=None, t_max=10000, epsilon=1e-3, lr=0.01, beta_2=0.999):

    beta_1 = 0.9
    beta_2 = 0.999  # initialize the values of the parameters
    # epsilon = 1e-3

    log_alpha = log_alpha0
    # log_alpha0 = 0  # initialize the vector
    m_t = 0
    v_t = 0
    t = 0

    for i in range(n_outer):
        t += 1
        val, grad = _get_val_grad(log_alpha)

        print("%i / %i  || crosss entropy %f  || accuracy %f" % (
              i, n_outer, val, monitor.acc_vals[-1]))
        # updates the moving averages of the gradient
        m_t = beta_1*m_t + (1-beta_1) * grad
        # updates the moving averages of the squared gradient
        v_t = beta_2*v_t + (1-beta_2) * (grad * grad)
        m_cap = m_t/(1-(beta_1**t))
        # calculates the bias-corrected estimates
        v_cap = v_t/(1-(beta_2**t))
        # calculates the bias-corrected estimates
        logh_alpha_prev = log_alpha
        # updates the parameters
        log_alpha = log_alpha - (lr*m_cap) / (np.sqrt(v_cap)+epsilon)
        # checks if it is converged or not
        if np.allclose(log_alpha, logh_alpha_prev):
            break


def grad_search_backtrack_dirty(
        criterion, log_alpha0, monitor, n_outer=10, warm_start=None, tol=1e-3,
        maxit_ln=10):

    def _get_val_grad(lambdak, tol=tol):
        return criterion.get_val_grad(lambdak, tol=tol, monitor=monitor)

    def _get_val(lambdak, tol=tol):
        return criterion.get_val(lambdak, tol=tol)

    log_alphak = log_alpha0
    for i in range(n_outer):
        val, grad = _get_val_grad(log_alphak)

        monitor(val.copy(), None, log_alphak, grad)

        # step_size = 1 / norm(grad)
        step_size = backtracking_ls(
            val, grad, _get_val, log_alphak, c=0.5, rho=0.5, maxit_ln=maxit_ln)
        log_alphak -= step_size * grad

        print("%i / %i  || crosss entropy %f  || accuracy %f" % (
            i, n_outer, val, monitor.acc_vals[-1]))


def grad_search_backtracking_cd_dirty(
        criterion, log_alpha0, monitor, n_outer=10, warm_start=None, tol=1e-3,
        maxit_ln=5):

    log_alpha = log_alpha0
    for i in range(n_outer):
        for k in range(log_alpha0.shape[0]):

            def _get_val(log_alphak, tol=tol):
                return criterion.get_val(log_alphanew, tol=tol)

            def _get_val_gradk(log_alphak, tol=tol):
                val, grad = criterion.get_val_grad(
                    log_alphanew, tol=tol, monitor=monitor)
                return val, grad[k]

            val, gradk = _get_val_gradk(log_alpha[k])

            step_size = backtracking_ls(
                val, gradk, _get_val, log_alpha[k], maxit_ln=maxit_ln)

            log_alpha[k] -= step_size * gradk

            print("%i / %i  || crosss entropy %f  || accuracy %f" % (
                i, n_outer, val, monitor.acc_vals[-1]))


def grad_search_backtracking_cd_dirty2(
        criterion, log_alpha0, monitor, n_outer=10, warm_start=None, tol=1e-3,
        maxit_ln=5):

    log_alpha = log_alpha0

    def _get_val(log_alphak, tol=tol):
        return criterion.get_valk(log_alphak, k, tol=tol)

    def _get_val_gradk(log_alpha, k, tol=tol):
        val, gradk = criterion.get_val_gradk(
            log_alpha, monitor=monitor, k=k, tol=tol)
        return val, gradk

    for i in range(n_outer):
        for k in range(log_alpha0.shape[0]):

            val, gradk = _get_val_gradk(log_alpha, k)

            step_size = backtracking_ls(
                val, gradk, _get_val, log_alpha[k], maxit_ln=maxit_ln)

            log_alpha[k] -= step_size * gradk

            print("%i / %i  || crosss entropy %f  || accuracy %f" % (
                i, n_outer, val, monitor.acc_vals[-1]))


def backtracking_ls(val, grad, _get_val, x, c=0.00001, rho=0.5, maxit_ln=50):
    alpha_ls = 1 / norm(grad)

    for i in range(maxit_ln):
        if _get_val(x - alpha_ls * grad) <= val - c * alpha_ls * norm(grad) ** 2:
            return alpha_ls
        else:
            alpha_ls = alpha_ls * rho

    print("WARNING no step found")
    return 0
    # return alpha_ls
