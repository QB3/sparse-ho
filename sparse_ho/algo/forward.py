import numpy as np
from scipy.sparse import issparse


class Forward():
    """Algorithm that will compute the (hyper)gradient, ie the gradient with
    respect to the hyperparameter using the forward algorithm.

    Parameters
    ----------
    verbose: bool
    """

    def __init__(self, verbose=False, use_stop_crit=True):
        self.verbose = verbose
        self.use_stop_crit = use_stop_crit

    def get_beta_jac_v(
            self, X, y, log_alpha, model, v, mask0=None, dense0=None,
            quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            compute_jac=True, full_jac_v=False):
        mask, dense, jac = get_beta_jac_iterdiff(
            X, y, log_alpha, model, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start,
            max_iter=max_iter, tol=tol,
            compute_jac=compute_jac, verbose=self.verbose,
            use_stop_crit=self.use_stop_crit)
        if jac is not None:
            jac_v = model.get_jac_v(X, y, mask, dense, jac, v)
            if full_jac_v:
                jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])
        else:
            jac_v = None
        return mask, dense, jac_v, jac


def get_beta_jac_iterdiff(
        X, y, log_alpha, model, mask0=None, dense0=None, jac0=None,
        max_iter=1000, tol=1e-3, compute_jac=True, return_all=False,
        save_iterates=False, verbose=False, use_stop_crit=True, gap_freq=10):
    """
    Parameters
    --------------
    X: np.array, shape (n_samples, n_features)
        design matrix
        It can also be a sparse CSC matrix
    y: np.array, shape (n_samples,)
        observations
    log_alpha: float or np.array, shape (n_features)
        log  of eth coefficient multiplying the penalization
    beta0: np.array, shape (n_features,)
        initial value of the regression coefficients
        beta for warm start
    dbeta0: np.array, shape (n_features,)
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
    model: string
        model used, "lasso", "wlasso", or "mcp"
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
    mask : np.array, shape (n_features,)
        The mask of non-zero coefficients in beta.
    dense : np.array, shape (n_nonzeros,)
        The beta coefficients on the support
    jac : np.array, shape (n_nonzeros,) or (n_nonzeros, q)
        The jacobian restricted to the support. If there are more than
        one hyperparameter then it has two dimensions.
    """
    n_samples, n_features = X.shape
    is_sparse = issparse(X)
    if not is_sparse and not np.isfortran(X):
        X = np.asfortranarray(X)
    L = model.get_L(X, is_sparse=is_sparse)

    ############################################
    alpha = np.exp(log_alpha)

    if hasattr(model, 'estimator') and model.estimator is not None:
        return model._use_estimator(X, y, alpha, tol, max_iter)

    try:
        alpha.shape[0]
        alphas = alpha.copy()
    except Exception:
        alphas = np.ones(n_features) * alpha
    ############################################
    # warm start for beta
    beta, r = model._init_beta_r(X, y, mask0, dense0)

    ############################################
    # warm start for dbeta
    dbeta, dr = model._init_dbeta_dr(
        X, y, mask0=mask0, dense0=dense0, jac0=jac0, compute_jac=compute_jac)

    # store the values of the objective
    pobj0 = model._get_pobj0(r, np.zeros(X.shape[1]), alphas, y)
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
                dbeta, r, dr, alphas, L, compute_jac=compute_jac)
        else:
            model._update_beta_jac_bcd(
                X, y, beta, dbeta, r, dr, alphas, L, compute_jac=compute_jac)

        pobj.append(model._get_pobj(r, X, beta, alphas, y))
        assert pobj[-1] >= 0

        if i > 1:
            if verbose:
                print("relative decrease = ", (pobj[-2] - pobj[-1]) / pobj0)

        if use_stop_crit and i % gap_freq == 0 and i > 0:
            if hasattr(model, "_get_dobj"):
                dobj = model._get_dobj(r, X, beta, alpha, y)
                dual_gap = pobj[-1] - dobj
                assert dual_gap >= -10 * np.finfo('float').eps
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

    if save_iterates:
        return np.array(list_beta), np.array(list_jac)
    if return_all:
        return mask, dense, list_beta
    else:
        if compute_jac:
            return mask, dense, jac
        else:
            return mask, dense, None
