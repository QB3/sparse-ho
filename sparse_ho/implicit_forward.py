import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
import scipy.sparse.linalg as slinalg
from sparse_ho.forward import get_beta_jac_iterdiff


class ImplicitForward():
    def __init__(self, criterion, tol_jac, n_iter=100, n_iter_jac=100):
        self.criterion = criterion
        self.n_iter = n_iter
        self.tol_jac = tol_jac
        self.n_iter_jac = n_iter_jac

    def get_beta_jac_v(
            self, X, y, log_alpha, model, get_v, mask0=None, dense0=None,
            quantity_to_warm_start=None, max_iter=1000, tol=1e-3,
            compute_jac=False, backward=False, full_jac_v=False):
        """TODO use get v function of the criterion
        """
        mask, dense, jac = get_beta_jac_fast_iterdiff(
            X, y, log_alpha, self.criterion.X_val, self.criterion.y_val,
            get_v, mask0=mask0, dense0=dense0,
            jac0=quantity_to_warm_start, tol_jac=self.tol_jac,
            tol=tol, niter_jac=self.n_iter_jac, model=model,
            max_iter=self.criterion.model.max_iter)
        jac_v = jac.T @ get_v(mask, dense)
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


def get_beta_jac_fast_iterdiff(
        X, y, log_alpha, X_val, y_val, get_v, model, mask0=None, dense0=None,
        jac0=None,
        tol=1e-3, max_iter=100, niter_jac=1000, tol_jac=1e-6):
    n_samples, n_features = X.shape
    mask, dense, _ = get_beta_jac_iterdiff(
        X, y, log_alpha, mask0=mask0, dense0=dense0, jac0=jac0, tol=tol,
        max_iter=max_iter, compute_jac=False, model=model)

    dbeta0_new = model._init_dbeta0(mask, mask0, jac0)
    reduce_alpha = model._reduce_alpha(np.exp(log_alpha), mask)

    v = get_v(mask, dense)
    r = X[:, mask] @ dense
    jac = get_only_jac(
        X[:, mask], r, reduce_alpha, np.sign(dense), v,
        dbeta=dbeta0_new, niter_jac=niter_jac, tol_jac=tol_jac, model=model,
        mask=mask, dense=dense)

    return mask, dense, jac


def get_only_jac(
        Xs, r, alpha, sign_beta, v, dbeta=None, niter_jac=100, tol_jac=1e-4,
        model="lasso", mask=None, dense=None):
    n_samples, n_features = Xs.shape

    is_sparse = issparse(Xs)
    if is_sparse:
        L = slinalg.norm(Xs, axis=0) ** 2 / n_samples
    else:
        L = norm(Xs, axis=0) ** 2 / n_samples

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
    dbeta_old = dbeta.copy()

    tol_crit = tol_jac * norm(v)
    dr = - Xs @ dbeta
    for i in range(niter_jac):
        print("%i -st iterations over %i" % (i, niter_jac))
        if is_sparse:
            model._update_only_jac_sparse(
                Xs.data, Xs.indptr, Xs.indices, n_samples,
                n_features, dbeta, r, dr, L, alpha, sign_beta)
        else:
            model._update_only_jac(Xs, r, dbeta, dr, L, alpha, sign_beta)

        # if model == "lasso":
        #     if is_sparse:
        #         _update_only_jac_sparse(
        #             Xs.data, Xs.indptr, Xs.indices, n_samples,
        #             n_features, dbeta, dr, L, alpha, sign_beta)
        #     else:
        #         _update_only_jac(Xs, dbeta, dr, L, alpha, sign_beta)
        # elif model == "mcp":
        #     if is_sparse:
        #         _update_only_jac_mcp_sparse(
        #             Xs.data, Xs.indptr, Xs.indices, n_samples, n_features,
        #             dense, dbeta, dr, alpha[0], alpha[1], L, compute_jac=True)
        #     else:
        #         _update_only_jac_mcp(
        #             Xs, dense, dbeta, dr, alpha[0], alpha[1],
        #             L, compute_jac=True)
        # elif model == "wlasso":
        #     if is_sparse:
        #         _update_only_jac_sparse_p(
        #             Xs.data, Xs.indptr, Xs.indices, n_samples,
        #             n_features, dbeta, dr, L, alpha[mask], sign_beta)
        #             # n_features, dbeta, dr, L, alpha, sign_beta)
        #     else:
        #         _update_only_jac_p(Xs, dbeta, dr, L, alpha[mask], sign_beta)
        print(norm(dbeta - dbeta_old))
        if norm(v @ (dbeta - dbeta_old)) < tol_crit:
            break
        dbeta_old = dbeta.copy()
    return dbeta
