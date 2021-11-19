import numpy as np
from scipy.sparse import issparse
from sparse_ho.algo.forward import compute_beta


class ImplicitForward():
    """Algorithm to compute the hypergradient using implicit forward
    differentiation.

    First the algorithm computes the regression coefficients.
    Then the iterations of the forward differentiation are applied to compute
    the Jacobian.

    Parameters
    ----------
    tol_jac: float
        Tolerance for the Jacobian computation.
    max_iter: int
        Maximum number of iterations for the inner solver.
    n_iter_jac: int
        Maximum number of iterations for the Jacobian computation.
    use_stop_crit: bool, optional (default=True)
        Use stopping criterion in hypergradient computation. If False,
        run to maximum number of iterations.
    verbose: bool, optional (default=False)
        Verbosity of the algorithm.
    """

    def __init__(
            self, tol_jac=1e-3, max_iter=100, n_iter_jac=100,
            use_stop_crit=True, verbose=False):
        self.max_iter = max_iter
        self.tol_jac = tol_jac
        self.n_iter_jac = n_iter_jac
        self.use_stop_crit = use_stop_crit
        self.verbose = verbose

    def get_beta_jac(
            self, X, y, log_alpha, model, get_grad_outer, mask0=None,
            dense0=None, quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            full_jac_v=False):
        """Compute beta and hypergradient using implicit forward
        differentiation.

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

        mask, dense, jac = get_bet_jac_implicit_forward(
            X, y, log_alpha, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start,
            tol_jac=tol, tol=tol, niter_jac=self.n_iter_jac, model=model,
            max_iter=self.max_iter, verbose=self.verbose)
        return mask, dense, jac

    def compute_beta_grad(
            self, X, y, log_alpha, model, get_grad_outer, mask0=None,
            dense0=None, quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            full_jac_v=False):
        mask, dense, jac = get_bet_jac_implicit_forward(
            X, y, log_alpha, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start,
            tol_jac=self.tol_jac, tol=tol, niter_jac=self.n_iter_jac,
            model=model, max_iter=self.max_iter, verbose=self.verbose,
            use_stop_crit=self.use_stop_crit)
        jac_v = model.get_jac_v(X, y, mask, dense, jac, get_grad_outer)
        if full_jac_v:
            jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])

        return mask, dense, jac_v, jac


def get_bet_jac_implicit_forward(
        X, y, log_alpha, model, mask0=None, dense0=None, jac0=None,
        tol=1e-3, max_iter=1000, niter_jac=1000, tol_jac=1e-6, verbose=False,
        use_stop_crit=True):

    mask, dense, _ = compute_beta(
        X, y, log_alpha, mask0=mask0, dense0=dense0, jac0=jac0, tol=tol,
        max_iter=max_iter, compute_jac=False, model=model, verbose=verbose,
        use_stop_crit=use_stop_crit)
    dbeta0_new = model._init_dbeta0(mask, mask0, jac0)
    reduce_alpha = model._reduce_alpha(np.exp(log_alpha), mask)

    _, dual_var = model._init_beta_dual_var(X, y, mask, dense)
    jac = get_only_jac(
        model.reduce_X(X, mask), model.reduce_y(y, mask), dual_var,
        reduce_alpha, model.sign(dense, log_alpha), dbeta=dbeta0_new,
        niter_jac=niter_jac, tol_jac=tol_jac, model=model, mask=mask,
        dense=dense, verbose=verbose, use_stop_crit=use_stop_crit)

    return mask, dense, jac


def get_only_jac(
        Xs, y, dual_var, alpha, sign_beta, dbeta=None, niter_jac=100,
        tol_jac=1e-4, model="lasso", mask=None, dense=None, verbose=False,
        use_stop_crit=True):
    n_samples, n_features = Xs.shape

    L = model.get_L(Xs)

    residual_norm = []

    if hasattr(model, 'dual'):
        ddual_var = model._init_ddual_var(dbeta, Xs, y, sign_beta, alpha)
        dbeta = model.dbeta
    else:
        if dbeta is None:
            dbeta = model._init_dbeta(n_features)
        ddual_var = model._init_ddual_var(dbeta, Xs, y, sign_beta, alpha)

    for i in range(niter_jac):
        if verbose:
            print("%i -st iterations over %i" % (i, niter_jac))
        if issparse(Xs):
            model._update_only_jac_sparse(
                Xs.data, Xs.indptr, Xs.indices, y, n_samples,
                n_features, dbeta, dual_var, ddual_var, L, alpha, sign_beta)
        else:
            model._update_only_jac(
                Xs, y, dual_var, dbeta, ddual_var, L, alpha, sign_beta)
        residual_norm.append(
            model.get_jac_residual_norm(
                Xs, y, n_samples, sign_beta, dbeta, dual_var,
                ddual_var, alpha))
        if use_stop_crit and i > 1:
            # relative stopping criterion for the computation of the jacobian
            # and absolute stopping criterion to handle warm start
            rel_tol = np.abs(residual_norm[-2] - residual_norm[-1])
            if (rel_tol < np.abs(residual_norm[-1]) * tol_jac
                    or residual_norm[-1] < 1e-10):
                break
    # HACK we only need this for one test, do not rely on it
    get_only_jac.n_iter = i

    return dbeta
