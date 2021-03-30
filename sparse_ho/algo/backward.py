import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from sparse_ho.algo.forward import compute_beta


class Backward():
    """Algorithm to compute the hypergradient using backward differentiation.

    The algorithm first computes the regression coefficients beta, using
    proximal coordinate descent, storing all the iterates.
    Then the gradient is computed in a backward way.

    Parameters
    ----------
    use_stop_crit: bool, optional (default=True)
        Use stopping criterion in hypergradient computation. If False,
        run to maximum number of iterations.
    verbose: bool, optional (default=False)
        Verbosity of the algorithm.
    """

    def __init__(self, use_stop_crit=True, verbose=False):
        self.use_stop_crit = use_stop_crit
        self.verbose = verbose

    def compute_beta_grad(
            self, X, y, log_alpha, model, get_grad_outer, mask0=None,
            dense0=None, quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            full_jac_v=False):
        """Compute beta and hypergradient with backward differentiation of
        proximal coordinate descent.

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
            Previous Jacobian of the inner optimization problem.
        max_iter: int
            Maximum number of iteration for the inner solver.
        tol: float
            The tolerance for the inner optimization problem.
        full_jac_v: bool
            TODO
        """

        # 1 compute the regression coefficients beta
        mask, dense, list_sign = compute_beta(
            X, y, log_alpha, model, mask0=mask0, dense0=dense0,
            jac0=None, max_iter=max_iter, tol=tol,
            compute_jac=False, return_all=True,
            use_stop_crit=self.use_stop_crit)
        v = np.zeros(X.shape[1])
        v[mask] = get_grad_outer(mask, dense)
        # 2 compute the gradient in a backward way
        grad = get_grad_backward(
            X, np.exp(log_alpha), list_sign, v, model,
            jac_v0=quantity_to_warm_start)

        if not full_jac_v:
            grad = model.get_mask_jac_v(mask, grad)

        grad = np.atleast_1d(grad)
        return mask, dense, grad, grad


def get_grad_backward(X, alpha, list_beta, v, model, jac_v0=None):
    n_samples, n_features = X.shape
    is_sparse = issparse(X)
    if is_sparse:
        L = slinalg.norm(X, axis=0) ** 2 / n_samples
    else:
        L = norm(X, axis=0) ** 2 / n_samples
    v_ = v.copy()
    list_beta = np.asarray(list_beta)
    grad = model._init_g_backward(None, n_features)
    for k in (np.arange(list_beta.shape[0] - 1, -1, -1)):
        beta = list_beta[k, :]
        if is_sparse:
            grad = model._update_bcd_jac_backward_sparse(
                X.data, X.indptr, X.indices, n_samples, n_features,
                alpha, grad, beta, v_, L)
        else:
            grad = model._update_bcd_jac_backward(
                X, alpha, grad, beta, v_, L)

    return grad
