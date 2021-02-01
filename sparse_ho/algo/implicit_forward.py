import numpy as np
from scipy.sparse import issparse
from sparse_ho.algo.forward import get_beta_jac_iterdiff


class ImplicitForward():
    """Algorithm that will compute the (hyper)gradient, ie the gradient with
    respect to the hyperparameter using the implicit forward algorithm.

    Parameters
    ----------
    max_iter: int
        maximum number of iteration for the inner solver
    tol_jac: float
        tolerance for the Jacobian computation
    n_iter_jac: int
        maximum number of iteration for the Jacobian computation
    verbose: bool
    """

    def __init__(
            self, tol_jac=1e-3, max_iter=100, n_iter_jac=100,
            verbose=False, use_stop_crit=True):
        self.max_iter = max_iter
        self.tol_jac = tol_jac
        self.n_iter_jac = n_iter_jac
        self.verbose = verbose
        self.use_stop_crit = use_stop_crit

    def get_beta_jac(
            self, X, y, log_alpha, model, get_v, mask0=None, dense0=None,
            quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            compute_jac=False, backward=False, full_jac_v=False):
        mask, dense, jac = get_beta_jac_fast_iterdiff(
            X, y, log_alpha, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start,
            # tol_jac=self.tol_jac,
            tol_jac=tol, tol=tol, niter_jac=self.n_iter_jac, model=model,
            max_iter=self.max_iter, verbose=self.verbose)
        return mask, dense, jac

    def get_beta_jac_v(
            self, X, y, log_alpha, model, get_v, mask0=None, dense0=None,
            quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            compute_jac=False, backward=False, full_jac_v=False):
        mask, dense, jac = get_beta_jac_fast_iterdiff(
            X, y, log_alpha, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start,
            # tol_jac=self.tol_jac,
            tol_jac=self.tol_jac, tol=tol, niter_jac=self.n_iter_jac,
            model=model, max_iter=self.max_iter, verbose=self.verbose,
            use_stop_crit=self.use_stop_crit)
        jac_v = model.get_jac_v(X, y, mask, dense, jac, get_v)
        if full_jac_v:
            jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])

        return mask, dense, jac_v, jac


def get_beta_jac_fast_iterdiff(
        X, y, log_alpha, model, mask0=None, dense0=None, jac0=None,
        tol=1e-3, max_iter=1000, niter_jac=1000, tol_jac=1e-6, verbose=False,
        use_stop_crit=True):

    mask, dense, _ = get_beta_jac_iterdiff(
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

    is_sparse = issparse(Xs)
    L = model.get_L(Xs, is_sparse)

    objs = []

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
        if is_sparse:
            model._update_only_jac_sparse(
                Xs.data, Xs.indptr, Xs.indices, y, n_samples,
                n_features, dbeta, dual_var, ddual_var, L, alpha, sign_beta)
        else:
            model._update_only_jac(
                Xs, y, dual_var, dbeta, ddual_var, L, alpha, sign_beta)
        objs.append(
            model.get_jac_obj(Xs, y, n_samples, sign_beta, dbeta, dual_var,
                              ddual_var, alpha))
        if use_stop_crit and i > 1:
            if np.abs(objs[-2] - objs[-1]) < np.abs(objs[-1]) * tol_jac:
                break
    return dbeta
