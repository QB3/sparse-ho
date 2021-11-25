import numpy as np

from scipy.sparse.linalg import cg

from sparse_ho.utils import init_dbeta0_new
from sparse_ho.algo.forward import compute_beta


class Implicit():
    """Algorithm to compute the hypergradient using implicit differentiation.

    First the algorithm computes the regression coefficients beta, then the
    gradient is computed after resolution of a linear system on the generalized
    support of beta.

    Parameters
    ----------
    max_iter: int (default=100)
        Maximum number of iteration for the inner solver.
    max_iter_lin_sys: int (default=100)
        Maximum number of iteration for the resolution of the linear system.
    tol_lin_sys: float (default=1e-6)
        Tolerance for the resolution of the linear system.
    """

    def __init__(self, max_iter=100, max_iter_lin_sys=100, tol_lin_sys=1e-6):
        self.max_iter = max_iter
        self.max_iter_lin_sys = max_iter_lin_sys
        self.tol_lin_sys = tol_lin_sys

    def compute_beta_grad(
            self, X, y, log_alpha, model, get_grad_outer, mask0=None,
            dense0=None, quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            full_jac_v=False):
        """Compute beta and the hypergradient, with implicit differentiation.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array, shape (n_features,)
            Logarithm of hyperparameter.
        model:  instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        get_grad_outer: callable
            Function which returns the gradient of the outer criterion.
        mask0: ndarray, shape (n_features,)
            Boolean of active feature of the previous regression coefficients
            beta for warm start.
        dense0: ndarray, shape (mask.sum(),)
            Initial value of the previous regression coefficients
            beta for warm start.
        quantity_to_warm_start: ndarray
            Previous solution of the linear system.
        max_iter: int
            Maximum number of iteration for the inner solver.
        tol: float
            The tolerance for the inner optimization problem.
        full_jac_v: bool
            TODO
        """
        mask, dense, jac_v, sol_lin_sys = compute_beta_grad_implicit(
            X, y, log_alpha, get_grad_outer, mask0=mask0, dense0=dense0,
            max_iter=max_iter, tol=tol, sol_lin_sys=quantity_to_warm_start,
            tol_lin_sys=self.tol_lin_sys,
            max_iter_lin_sys=self.max_iter_lin_sys, model=model)

        if full_jac_v:
            jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])

        return mask, dense, jac_v, sol_lin_sys


def compute_beta_grad_implicit(
        X, y, log_alpha, get_grad_outer, mask0=None, dense0=None, tol=1e-3,
        model="lasso", max_iter=1000, sol_lin_sys=None,
        tol_lin_sys=1e-6, max_iter_lin_sys=100):
    """Compute beta and the hypergradient with implicit differentiation.

    The hypergradient computation is done in 3 steps:
    - 1 solve the inner optimization problem.
    - 2 solve a linear system on the support (ie the non-zeros coefficients)
    of the solution.
    - 3 use the solution of the linear system to compute the gradient.

    Parameters
    ----------
    X: array-like, shape (n_samples, n_features)
        Design matrix.
    y: ndarray, shape (n_samples,)
        Observation vector.
    log_alpha: float or np.array, shape (n_features,)
        Logarithm of hyperparameter.
    mask0: ndarray, shape (n_features,)
        Boolean of active feature of the previous regression coefficients
        beta for warm start.
    dense0: ndarray, shape (mask.sum(),)
        Initial value of the previous regression coefficients
        beta for warm start.
    tol: float
        The tolerance for the inner optimization problem.
    model:  instance of ``sparse_ho.base.BaseModel``
        A model that follows the sparse_ho API.
    max_iter: int
        Maximum number of iterations for the inner solver.
    sol_lin_sys: ndarray
        Previous solution of the linear system for warm start.
    tol_lin_sys: float
        Tolerance for the resolution of the linear system.
    max_iter_lin_sys: int
        Maximum number of iterations for the resolution of the linear system.
    """

    # 1 compute the regression coefficients beta, stored in mask and dense
    alpha = np.exp(log_alpha)
    mask, dense, _ = compute_beta(
        X, y, log_alpha, mask0=mask0, dense0=dense0,
        tol=tol, max_iter=max_iter, compute_jac=False, model=model)
    n_features = X.shape[1]

    mat_to_inv = model.get_mat_vec(X, y, mask, dense, log_alpha)

    v = get_grad_outer(mask, dense)
    if hasattr(model, 'dual'):
        v = model.get_dual_v(mask, dense, X, y, v, log_alpha)

    # 2 solve the linear system
    # TODO I think this should be removed
    if not alpha.shape:
        alphas = np.ones(n_features) * alpha
    else:
        alphas = alpha.copy()
    if sol_lin_sys is not None and not hasattr(model, 'dual'):
        sol0 = init_dbeta0_new(sol_lin_sys, mask, mask0)
    else:
        sol0 = None  # TODO add warm start for SVM and SVR
    sol = cg(
        mat_to_inv, - model.generalized_supp(X, v, log_alpha),
        x0=sol0, tol=tol_lin_sys, maxiter=max_iter_lin_sys)
    sol_lin_sys = sol[0]

    # 3 compute the gradient
    grad = model._get_grad(X, y, sol_lin_sys, mask, dense, alphas, v)
    return mask, dense, grad, sol_lin_sys
