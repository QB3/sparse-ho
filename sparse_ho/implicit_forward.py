import numpy as np
from scipy.sparse import issparse
from sparse_ho.forward import get_beta_jac_iterdiff


class ImplicitForward():
    def __init__(
            self, criterion, tol_jac=1e-3, n_iter=100, n_iter_jac=100,
            use_sk=False, verbose=False):
        self.criterion = criterion
        self.n_iter = n_iter
        self.tol_jac = tol_jac
        self.n_iter_jac = n_iter_jac
        self.use_sk = use_sk
        self.verbose = verbose

    def get_beta_jac_v(
            self, X, y, log_alpha, model, get_v, mask0=None, dense0=None,
            quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            compute_jac=False, backward=False, full_jac_v=False):
        mask, dense, jac = get_beta_jac_fast_iterdiff(
            X, y, log_alpha, self.criterion.X_val, self.criterion.y_val,
            get_v, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start,
            # tol_jac=self.tol_jac,
            tol_jac=tol, use_sk=self.use_sk,
            tol=tol, niter_jac=self.n_iter_jac, model=model,
            max_iter=self.criterion.model.max_iter, verbose=self.verbose)
        jac_v = model.get_jac_v(mask, dense, jac, get_v)
        if full_jac_v:
            jac_v = model.get_full_jac_v(mask, jac_v, X.shape[1])
        return mask, dense, jac_v, jac

    def get_val_grad(
            self, log_alpha, mask0=None, dense0=None, beta_star=None,
            jac0=None, max_iter=1000, tol=1e-3, compute_jac=True,
            backward=False):
        return self.criterion.get_val_grad(
            log_alpha, self.get_beta_jac_v, max_iter=max_iter, tol=tol,
            compute_jac=compute_jac, backward=backward)

    def get_val(
            self, log_alpha, mask0=None, dense0=None, beta_star=None,
            jac0=None, max_iter=1000, tol=1e-3, compute_jac=True,
            backward=False):
        return self.criterion.get_val(
            log_alpha, self.get_beta_jac_v, max_iter=max_iter, tol=tol,
            compute_jac=compute_jac, backward=backward)


def get_beta_jac_fast_iterdiff(
        X, y, log_alpha, X_val, y_val, get_v, model, mask0=None, dense0=None, jac0=None, tol=1e-3, max_iter=1000, niter_jac=1000, tol_jac=1e-6, use_sk=False, verbose=False):
    n_samples, n_features = X.shape

    mask, dense, _ = get_beta_jac_iterdiff(
        X, y, log_alpha, mask0=mask0, dense0=dense0, jac0=jac0, tol=tol,
        max_iter=max_iter, compute_jac=False, model=model, use_sk=use_sk,
        verbose=verbose)

    dbeta0_new = model._init_dbeta0(mask, mask0, jac0)
    reduce_alpha = model._reduce_alpha(np.exp(log_alpha), mask)

    v = None
    _, r = model._init_beta_r(X, y, mask, dense)
    jac = get_only_jac(
        model.reduce_X(mask), model.reduce_y(mask), r, reduce_alpha, model.sign(dense), v,
        dbeta=dbeta0_new, niter_jac=niter_jac, tol_jac=tol_jac, model=model, mask=mask, dense=dense, verbose=verbose)

    return mask, dense, jac


def get_only_jac(
        Xs, y, r, alpha, sign_beta, v, dbeta=None, niter_jac=100, tol_jac=1e-4, model="lasso", mask=None, dense=None, verbose=False):
    n_samples, n_features = Xs.shape

    is_sparse = issparse(Xs)
    L = model.get_L(Xs, is_sparse)

    objs = []

    if dbeta is None:
        model._init_dbeta(n_features)
        # if model == "lasso":
        #     dbeta = np.zeros(n_features)
        # if model == "mcp":
        #     dbeta = np.zeros((n_features, 2))
        # elif model == "wlasso":
        #     dbeta = np.zeros((n_features, n_features))
    else:
        dbeta = dbeta.copy()

    dr = model._init_dr(dbeta, Xs, y, mask)
    for i in range(niter_jac):
        if verbose:
            print("%i -st iterations over %i" % (i, niter_jac))
        if is_sparse:
            model._update_only_jac_sparse(
                Xs.data, Xs.indptr, Xs.indices, y, n_samples,
                n_features, dbeta, r, dr, L, alpha, sign_beta)
        else:
            model._update_only_jac(
                Xs, y, r, dbeta, dr, L, alpha, sign_beta, mask)

        objs.append(
            model.get_jac_obj(Xs, y, sign_beta, dbeta, r, dr, alpha, mask))

        # m1 = norm(- v.T @ Xs.T @ dr + sign_beta * n_samples * alpha)
        # m2 = tol_jac * np.sqrt(n_features) * n_samples * alpha * norm(v)
        # crit = m1 <= m2
        # print("m1 %.2f", m1)
        # print("m2 %.2f", m2)
        # print("m1 = %f" % norm(v @ (dbeta - dbeta_old)))
        # print("tol_crit %f" % tol_crit)
        # if norm(v @ (dbeta - dbeta_old)) < tol_crit:
        # if norm((dbeta - dbeta_old)) < tol_jac * norm(dbeta):
        # crit =
        print('jac obj', objs[-1])
        if i > 1 and np.abs(objs[-2] - objs[-1]) < np.abs(objs[-1]) * tol_jac:
            break
        # dbeta_old = dbeta.copy()
        # dr_old = dr.copy()

    return dbeta
