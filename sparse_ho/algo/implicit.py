import numpy as np
from numpy.linalg import norm

from scipy.sparse import issparse, identity
from scipy.sparse.linalg import cg

from sparse_ho.utils import init_dbeta0_new
from sparse_ho.algo.forward import get_beta_jac_iterdiff


class Implicit():
    """Algorithm that will compute the (hyper)gradient, ie the gradient with
    respect to the hyperparameter using the implicit differentiation.

    Parameters
    ----------
    max_iter: int
            maximum number of iteration for the inner solver
    """

    def __init__(self, max_iter=100):
        self.max_iter = max_iter

    def get_beta_jac_v(
            self, X, y, log_alpha, model, get_v, mask0=None, dense0=None,
            jac0=None, quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            backward=False, full_jac_v=False):

        mask, dense, jac_v, sol_lin_sys = get_beta_jac_t_v_implicit(
            X, y, log_alpha, get_v,
            mask0=mask0, dense0=dense0, max_iter=max_iter,
            sol_lin_sys=quantity_to_warm_start, tol=tol, model=model)

        if full_jac_v:
            jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])

        return mask, dense, jac_v, sol_lin_sys


def get_beta_jac_t_v_implicit(
        X, y, log_alpha, get_v, mask0=None, dense0=None, tol=1e-3,
        model="lasso", sk=False, max_iter=1000, sol_lin_sys=None,
        tol_lin_sys=1e-6, max_iter_lin_sys=100):
    alpha = np.exp(log_alpha)
    mask, dense, _ = get_beta_jac_iterdiff(
        X, y, log_alpha, mask0=mask0, dense0=dense0,
        tol=tol, max_iter=max_iter, compute_jac=False, model=model)
    n_features = X.shape[1]

    mat_to_inv = model.get_mv(X, y, mask, dense, log_alpha)

    v = get_v(mask, dense)
    if hasattr(model, 'dual'):
        v = model.get_dual_v(mask, dense, X, y, v, log_alpha)

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

    jac_t_v = model._get_jac_t_v(
        X, y, sol_lin_sys, mask, dense, alphas, v.copy())
    return mask, dense, jac_t_v, sol[0]
