# This files contains the functions for the Lasso, MCP and adaptive lasso
# hyperparameter setting

import time
import numpy as np
from numpy.linalg import norm

from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
from sparse_ho.implicit import get_beta_jac_t_v_implicit
from sparse_ho.backward import get_beta_jac_backward
from sparse_ho.cvxpylayers import get_beta_jac_cvxpy


def grad_search(
        X_train, y_train, log_alpha0, X_val, y_val, X_test, y_test,
        tol, monitor, method="implicit", maxit=1000, n_outer=100,
        warm_start=None, verbose=True,
        tolerance_decrease='constant', niter_jac=100, tol_jac=1e-3,
        model="lasso", convexify=False, gamma=1e-2, criterion="cv", sigma=1,
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

    if model == "lasso":
        isinstance(1.0, float)
    if model == "wlasso":
        assert log_alpha0.shape[0] == X_train.shape[1]
    if model == "mcp":
        assert log_alpha0.shape[0] == 2

    n_samples = X_train.shape[0]
    alpha_max = np.abs((X_train.T @ y_train)).max() / n_samples
    log_alpha_max = np.log(alpha_max)

    lambdak = log_alpha0.copy()
    old_lambdak = lambdak.copy()
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
            old_tol = seq_tol[i-1]
        except:
            old_tol = seq_tol[0]
        g_func, grad_lambda = get_val_grad(
            X_train, y_train, lambdak, X_val, y_val,
            X_test, y_test, tol, monitor,
            method=method, maxit=maxit, warm_start=warm_start,
            niter_jac=niter_jac, tol_jac=tol_jac, model=model,
            convexify=convexify, gamma=gamma,
            criterion=criterion, sigma=sigma, beta_star=beta_star)

        if monitor.times[-1] > t_max:
            break

        old_grads.append(norm(grad_lambda))
        if np.isnan(old_grads[-1]):
            print("Nan present in gradient")
            break

        if L_lambda is None:
            if old_grads[-1] > 1e-3:
                # make sure we are not selecting a step size that is too small
                try:
                    L_lambda = old_grads[-1] / np.sqrt(len(lambdak))
                except:
                    L_lambda = old_grads[-1]
            else:
                L_lambda = 1
        step_size = (1./L_lambda)

        if model == "lasso":
            old_lambdak = lambdak
        else:
            old_lambdak = lambdak.copy()
        lambdak -= step_size * grad_lambda
        if model == "mcp":
            assert lambdak.shape[0] == 2

        incr = norm(step_size * grad_lambda)

        C = 0.25
        factor_L_lambda = 1.0
        if g_func <= g_func_old + C * tol + \
                old_tol * (C + factor_L_lambda) * incr - factor_L_lambda * \
                (L_lambda) * incr * incr:
            L_lambda *= 0.95
            if verbose > 1:
                print('increased step size')
            lambdak -= step_size * grad_lambda
            if model == "mcp":
                assert lambdak.shape[0] == 2
        elif g_func >= 1.2 * g_func_old:
            if verbose > 1:
                print('decrease step size')
            # decrease step size
            L_lambda *= 2
            lambdak = old_lambdak.copy()
            print('!!step size rejected!!', g_func, g_func_old)
            g_func_old, _ = get_val_grad(
                X_train, y_train, old_lambdak, X_val, y_val,
                X_test, y_test, tol, monitor,
                method=method, maxit=maxit, warm_start=warm_start,
                niter_jac=niter_jac, tol_jac=tol_jac, model=model,
                convexify=convexify, gamma=gamma, criterion=criterion,
                sigma=sigma, beta_star=beta_star)

            # g_func_grad(x, old_lambdak)
            # tighten tolerance
            tol *= 0.5
        else:
            old_lambdak = lambdak.copy()
            lambdak -= step_size * grad_lambda
        if model == "lasso":
            if lambdak < -12:
                # lambdak = - 12.0
                print("alpha k smaller than -12")
            elif lambdak > log_alpha_max + np.log(0.9):
                lambdak = log_alpha_max + np.log(0.9)
        elif model == "mcp":
            assert lambdak.shape[0] == 2
            if lambdak[0] > log_alpha_max + np.log(0.9):
                lambdak[0] = log_alpha_max + np.log(0.9)
            if lambdak[1] <= 1.001:
                lambdak[1] = 1.001
        else:
            # lambdak[lambdak < -12] = -12
            lambdak[lambdak > log_alpha_max] = log_alpha_max
        g_func_old = g_func

    return lambdak, g_func, grad_lambda


def get_val_grad(
        X_train, y_train, log_alpha, X_val, y_val, X_test, y_test, tol,
        monitor, warm_start, method="implicit", maxit=1000,
        niter_jac=1000, model="lasso", tol_jac=1e-3, convexify=False,
        gamma=1e-2, criterion="cv", C=2.0, gamma_sure=0.3, sigma=1,
        random_state=42, beta_star=None):
    """
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

    n_samples, n_features = X_train.shape

    # warm start for cross validation loss and sure
    mask0, dense0, jac0, mask20, dense20, jac20 = (
        warm_start.mask_old, warm_start.beta_old, warm_start.dbeta_old,
        warm_start.mask_old2, warm_start.beta_old2, warm_start.dbeta_old2)

    if criterion == "cv":
        mask2 = None
        dense2 = None
        jac2 = None
        rmse = None
        if method == "implicit":
            sol_lin_sys = warm_start.sol_lin_sys
            mask, dense, jac, sol_lin_sys = get_beta_jac_t_v_implicit(
                X_train, y_train, log_alpha, X_val, y_val,
                mask0=mask0, dense0=dense0, jac0=jac0, tol=tol, model=model)
            warm_start.set_sol_lin_sys(sol_lin_sys)
        elif method == "forward":
            mask, dense, jac = get_beta_jac_iterdiff(
                X_train, y_train, log_alpha, mask0=mask0, dense0=dense0,
                jac0=jac0, tol=tol, maxit=maxit, model=model)

        elif method == "implicit_forward":
            mask, dense, jac = get_beta_jac_fast_iterdiff(
                X_train, y_train, log_alpha, X_val, y_val, mask0, dense0, jac0,
                tol, maxit, niter_jac=niter_jac, tol_jac=tol_jac, model=model)
        elif method == "backward":
            mask, dense, jac = get_beta_jac_backward(
                X_train, y_train, log_alpha, X_val, y_val, mask0=mask0,
                dense0=dense0, jac0=jac0, tol=tol, maxit=maxit)
        elif method == "hyperopt":
            mask, dense = get_beta_jac_iterdiff(
                X_train, y_train, log_alpha, mask0=mask0, dense0=dense0,
                tol=tol, maxit=maxit, model=model, compute_jac=False)
            jac = None
        elif method == "cvxpylayer":
            mask, dense, jac = get_beta_jac_cvxpy(
                X_train, y_train, log_alpha, X_val, y_val, mask0=mask0,
                dense0=dense0, model=model)
        else:
            raise ValueError('No method called %s' % method)
        # value of the objective function on the validation loss
        if convexify:
            val = norm(y_val - X_val[:, mask] @ dense) ** 2 / X_val.shape[0]
            val += gamma * np.sum(np.exp(log_alpha) ** 2)
        else:
            val = norm(y_val - X_val[:, mask] @ dense) ** 2 / X_val.shape[0]
        # value of the objective function on the test loss
        val_test = norm(
            y_test - X_test[:, mask] @ dense) ** 2 / X_test.shape[0]

        if method in (
                "implicit", "backward", "hyperopt", "cvxpylayer"):
            grad = jac
        else:
            if model in ("lasso", "mcp"):
                grad = 2 * jac.T @ (X_val[:, mask].T @ (
                    X_val[:, mask] @ dense - y_val)) / X_val.shape[0]

            elif model == "wlasso":
                grad = np.zeros(n_features)
                grad[mask] = 2 * jac.T @ (X_val[:, mask].T @ (
                    X_val[:, mask] @ dense - y_val)) / X_val.shape[0]
                if convexify:
                    grad += gamma * np.exp(log_alpha)
    elif criterion == "sure":
        val_test = 0
        epsilon = C * sigma / (n_samples) ** gamma_sure
        # TODO properly
        rng = np.random.RandomState(random_state)
        delta = rng.randn(n_samples)  # sample random noise for MCMC step
        y_train2 = y_train + epsilon * delta
        if method == "implicit":
            # TODO
            sol_lin_sys = warm_start.sol_lin_sys
            mask, dense, jac, sol_lin_sys = get_beta_jac_t_v_implicit(
                X_train, y_train, log_alpha, X_val, y_val,
                mask0=mask0, dense0=dense0, jac0=jac0, tol=tol, model=model,
                criterion="sure", n=1, sol_lin_sys=sol_lin_sys,
                sigma=sigma, epsilon=epsilon, delta=delta)
            sol_lin_sys2 = warm_start.sol_lin_sys2
            mask2, dense2, jac2, sol_lin_sys2 = get_beta_jac_t_v_implicit(
                X_train, y_train, log_alpha, X_val, y_val,
                mask0=mask20, dense0=dense20, jac0=jac20, tol=tol, model=model,
                criterion="sure", n=2, sol_lin_sys=sol_lin_sys2,
                sigma=sigma, epsilon=epsilon, delta=delta)
            warm_start.set_sol_lin_sys(sol_lin_sys)
            # 1 / 0
        elif method == "forward":
            mask, dense, jac = get_beta_jac_iterdiff(
                X_train, y_train, log_alpha, mask0=mask0, dense0=dense0,
                jac0=jac0, tol=tol, maxit=maxit, model=model)
            mask2, dense2, jac2 = get_beta_jac_iterdiff(
                X_train, y_train2, log_alpha, mask0=mask20, dense0=dense20,
                jac0=jac20, tol=tol, maxit=maxit, model=model)
        elif method == "implicit_forward":
            # TODO modify
            mask, dense, jac = get_beta_jac_fast_iterdiff(
                X_train, y_train, log_alpha, None, None, mask0, dense0, jac0,
                tol, maxit, criterion="sure", niter_jac=niter_jac,
                tol_jac=tol_jac, model=model,
                sigma=sigma, epsilon=epsilon, delta=delta, n=1)
            mask2, dense2, jac2 = get_beta_jac_fast_iterdiff(
                X_train, y_train2, log_alpha, X_val, y_val,
                mask20, dense20, jac20, tol, maxit, criterion="sure",
                niter_jac=niter_jac, tol_jac=tol_jac, model=model,
                sigma=sigma, epsilon=epsilon, delta=delta, n=2)
        elif method == "backward":
            mask, dense, jac = get_beta_jac_backward(
                X_train, y_train, log_alpha, X_val, y_val, mask0=mask0,
                dense0=dense0, jac0=jac0, tol=tol, maxit=maxit)
            mask2, dense2, jac2 = get_beta_jac_backward(
                X_train, y_train2, log_alpha, X_val, y_val,
                mask0=mask02, dense0=dense02, jac0=jac02, tol=tol, maxit=maxit)
        elif method == "hyperopt":
            mask, dense = get_beta_jac_iterdiff(
                X_train, y_train, log_alpha, mask0=mask0, dense0=dense0,
                tol=tol, maxit=maxit, model=model, compute_jac=False)
            mask2, dense2 = get_beta_jac_iterdiff(
                X_train, y_train2, log_alpha, mask0=mask20, dense0=dense20,
                tol=tol, maxit=maxit, model=model, compute_jac=False)
            jac, jac2 = None, None

        # compute the degree of freedom
        dof = (X_train[:, mask2] @ dense2 - X_train[:, mask] @ dense) @ delta
        dof /= epsilon
        # compute the value of the sure
        val = norm(y_train - X_train[:, mask] @ dense) ** 2
        val -= n_samples * sigma ** 2
        val += 2 * sigma ** 2 * dof
        if convexify:
            val += gamma * np.sum(np.exp(log_alpha) ** 2)

        if beta_star is not None:
            diff_beta = beta_star.copy()
            diff_beta[mask] -= dense
            rmse = norm(diff_beta)
        else:
            rmse = None

        if method == "hyperopt":
            monitor(val, None, log_alpha, rmse=rmse)
            return val
        if method in (
                "implicit", "backward", "hyperopt", "cvxpylayer"):
            grad = jac
        elif model == "lasso":
            grad = 2 * jac.T @ X_train[:, mask].T @ (
                X_train[:, mask] @ dense - y_train -
                delta * sigma ** 2 / epsilon)
            grad += (2 * sigma ** 2 *
                     jac2.T @ X_train[:, mask2].T @ delta / epsilon)
        elif model == "wlasso":
            grad = np.zeros(n_features)
            grad[mask] = 2 * jac.T @ X_train[:, mask].T @ (
                X_train[:, mask] @ dense -
                y_train - delta * sigma ** 2 / epsilon)
            grad[mask2] += (2 * sigma ** 2 *
                            jac2.T @ X_train[:, mask2].T @ delta / epsilon)
            if convexify:
                grad += gamma * np.exp(log_alpha)

    warm_start(mask, dense, jac, mask2, dense2, jac2)
    if model == "lasso":
        monitor(val, val_test, log_alpha, grad, rmse=rmse)
    elif model in ("mcp", "wlasso"):
        monitor(val, val_test, log_alpha.copy(), grad)
    else:
        monitor(val, val_test, log_alpha.copy(), rmse=rmse)
    if method == "hyperopt":
        return val
    else:
        return val, np.array(grad)
