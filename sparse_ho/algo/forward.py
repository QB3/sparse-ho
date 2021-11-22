import numpy as np
from scipy.sparse import issparse


class Forward():
    """Algorithm to compute the hypergradient using forward differentiation of
    proximal coordinate descent.

    The algorithm jointly and iteratively computes the regression coefficients
    and the Jacobian using forward differentiation of proximal
    coordinate descent.

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
        """Compute beta and hypergradient, with forward differentiation of
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
        # jointly compute the regression coefficients beta and the Jacobian
        mask, dense, jac = compute_beta(
            X, y, log_alpha, model, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start, max_iter=max_iter, tol=tol,
            compute_jac=True, verbose=self.verbose,
            use_stop_crit=self.use_stop_crit)
        if jac is not None:
            jac_v = model.get_jac_v(X, y, mask, dense, jac, get_grad_outer)
            if full_jac_v:
                jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])
        else:
            jac_v = None

        return mask, dense, jac_v, jac


def compute_beta(
        X, y, log_alpha, model, mask0=None, dense0=None, jac0=None,
        max_iter=1000, tol=1e-3, compute_jac=True, return_all=False,
        save_iterates=False, verbose=False, use_stop_crit=True, gap_freq=10):
    """
    Parameters
    --------------
    X: array-like, shape (n_samples, n_features)
        Design matrix.
    y: ndarray, shape (n_samples,)
        Observation vector.
    log_alpha: float or np.array, shape (n_features,)
        Logarithm of hyperparameter.
    beta0: ndarray, shape (n_features,)
        initial value of the regression coefficients
        beta for warm start
    dbeta0: ndarray, shape (n_features,)
        initial value of the jacobian dbeta for warm start
    max_iter: int
        number of iterations of the algorithm
    tol: float
        The tolerance for the optimization: if the updates are
        smaller than ``tol``, the optimization code checks the
        primal decrease for optimality and continues until it
        is smaller than ``tol``
    compute_jac: bool
        to compute or not the Jacobian along with the regression
        coefficients
    model:  instance of ``sparse_ho.base.BaseModel``
        A model that follows the sparse_ho API.
    return_all: bool
        to store the iterates or not in order to compute the Jacobian in a
        backward way
    use_stop_crit: bool
        use a stopping criterion or do all the iterations
    gap_freq : int
        After how many passes on the data the dual gap should be computed
        to stop the iterations.

    Returns
    -------
    mask : ndarray, shape (n_features,)
        The mask of non-zero coefficients in beta.
    dense : ndarray, shape (n_nonzeros,)
        The beta coefficients on the support
    jac : ndarray, shape (n_nonzeros,) or (n_nonzeros, q)
        The jacobian restricted to the support. If there are more than
        one hyperparameter then it has two dimensions.
    """
    n_samples, n_features = X.shape
    is_sparse = issparse(X)
    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)
    L = model.get_L(X)

    ############################################
    alpha = np.exp(log_alpha)

    if hasattr(model, 'estimator') and model.estimator is not None:
        return model._use_estimator(X, y, alpha, tol)

    try:
        alpha.shape[0]
        alphas = alpha.copy()
    except Exception:
        alphas = np.ones(n_features) * alpha
    ############################################
    # warm start for beta
    beta, dual_var = model._init_beta_dual_var(X, y, mask0, dense0)
    ############################################
    # warm start for dbeta
    dbeta, ddual_var = model._init_dbeta_ddual_var(
        X, y, mask0=mask0, dense0=dense0, jac0=jac0, compute_jac=compute_jac)

    # store the values of the objective
    pobj0 = model._get_pobj0(dual_var, np.zeros(X.shape[1]), alphas, y)
    pobj = []

    ############################################
    # store the iterates if needed
    if return_all:
        list_beta = []
    if save_iterates:
        list_beta = []
        list_jac = []

    for i in range(max_iter):
        if verbose:
            print("%i -st iteration over %i" % (i, max_iter))
        if is_sparse:
            model._update_beta_jac_bcd_sparse(
                X.data, X.indptr, X.indices, y, n_samples, n_features, beta,
                dbeta, dual_var, ddual_var, alphas, L,
                compute_jac=compute_jac)
        else:
            model._update_beta_jac_bcd(
                X, y, beta, dbeta, dual_var, ddual_var, alphas,
                L, compute_jac=compute_jac)

        pobj.append(model._get_pobj(dual_var, X, beta, alphas, y))

        if i > 1:
            if verbose:
                print("relative decrease = ", (pobj[-2] - pobj[-1]) / pobj0)

        if use_stop_crit and i % gap_freq == 0 and i > 0:
            if hasattr(model, "_get_dobj"):
                dobj = model._get_dobj(dual_var, X, beta, alpha, y)
                dual_gap = pobj[-1] - dobj
                if verbose:
                    print("dual gap %.2e" % dual_gap)
                if verbose:
                    print("gap %.2e" % dual_gap)
                if dual_gap < pobj0 * tol:
                    break
            else:
                if (pobj[-2] - pobj[-1] <= pobj0 * tol):
                    break
        if return_all:
            list_beta.append(beta.copy())
        if save_iterates:
            list_beta.append(beta.copy())
            list_jac.append(dbeta.copy())
    else:
        if verbose:
            print('did not converge !')

    mask = beta != 0
    dense = beta[mask]
    jac = model._get_jac(dbeta, mask)
    if hasattr(model, 'dual'):
        model.dual_var = dual_var
        if compute_jac:
            model.ddual_var = ddual_var
    if save_iterates:
        return np.array(list_beta), np.array(list_jac)
    if return_all:
        return mask, dense, list_beta
    else:
        if compute_jac:
            return mask, dense, jac
        else:
            return mask, dense, None
