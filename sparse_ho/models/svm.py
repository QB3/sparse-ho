import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg
from scipy.sparse import issparse

from numba import njit

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import proj_box_svm, ind_box, compute_grad_proj
from sparse_ho.utils import init_dbeta0_new


class SVM(BaseModel):
    """The support vector Machine classifier without bias
    The optimization problem is solved in the dual:
    1/2 r^T(y * X)(y * X)^T r - sum_i^n r_i
    s.t 0 <= r_i <= C

    Parameters
    ----------
    log_C : float
        logarithm of the hyperparameter C
    max_iter : int
        maximum number of epochs for the coordinate descent
        algorithm
    tol : float
        tolerance for the stopping criterion
    """

    def __init__(self, estimator=None, max_iter=100):
        self.estimator = estimator
        self.max_iter = max_iter
        self.dual = True
        self.dr = None

    def _init_dbeta_dr(self, X, y, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dr = np.ones(n_samples)
        if self.dr is not None:
            dr = self.dr.copy()
        if issparse(X):
            dbeta = (X.T).multiply(y * dr)
            dbeta = np.sum(dbeta, axis=1)
            dbeta = np.squeeze(np.array(dbeta))
        else:
            dbeta = np.sum(y * dr * X.T, axis=1)
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        r = np.zeros(n_samples)
        if mask0 is None or self.r is None:
            beta = np.zeros(n_features)
        else:
            r = self.r
            if issparse(X):
                beta = (X.T).multiply(y * r)
                beta = np.sum(beta, axis=1)
                beta = np.squeeze(np.array(beta))
            else:
                beta = np.sum(y * r * X.T, axis=1)
        return beta, r

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, C, L, compute_jac=True):
        """
            beta : dual variable of the svm
            r : primal used for cheap updates
            dbeta : jacobian of the dual variables
            dr : jacobian of the primal variable
        """
        C = C[0]
        n_samples = X.shape[0]
        for j in range(n_samples):
            F = y[j] * np.sum(beta * X[j, :]) - 1.0
            r_old = r[j]
            zj = r[j] - F / L[j]
            r[j] = proj_box_svm(zj, C)
            beta += (r[j] - r_old) * y[j] * X[j, :]
            if compute_jac:
                dF = y[j] * np.sum(dbeta * X[j, :])
                dr_old = dr[j]
                dzj = dr[j] - dF / L[j]
                dr[j] = ind_box(zj, C) * dzj
                dr[j] += C * (C <= zj)
                dbeta += (dr[j] - dr_old) * y[j] * X[j, :]

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, r, dr, C, L, compute_jac=True):
        # data needs to be a row sparse matrix
        non_zeros = np.where(L != 0)[0]
        C = C[0]
        for j in non_zeros:
            # get the i-st row of X in sparse format
            Xis = data[indptr[j]:indptr[j+1]]
            # get the non zero indices
            idx_nz = indices[indptr[j]:indptr[j+1]]

            # compute gradient_i
            G = y[j] * np.sum(beta[idx_nz] * Xis) - 1.0

            # compute projected gradient
            PG = compute_grad_proj(r[j], G, C)

            if np.abs(PG) > 1e-12:
                r_old = r[j]
                # update one coefficient SVM
                zj = r[j] - G / L[j]
                r[j] = min(max(zj, 0), C)
                beta[idx_nz] += (r[j] - r_old) * y[j] * Xis
                if compute_jac:
                    dr_old = dr[j]
                    dG = y[j] * np.sum(dbeta[idx_nz] * Xis)
                    dzj = dr[j] - dG / L[j]
                    dr[j:j+1] = ind_box(zj, C) * dzj
                    dr[j:j+1] += C * (C <= zj)
                    # update residuals
                    dbeta[idx_nz] += (dr[j] - dr_old) * y[j] * Xis

    @staticmethod
    def _get_pobj0(r, beta, C, y):
        C = C[0]
        n_samples = r.shape[0]
        obj_prim = C * np.sum(np.maximum(
            np.ones(n_samples), np.zeros(n_samples)))
        return obj_prim

    @staticmethod
    def _get_pobj(r, X, beta, C, y):
        C = C[0]
        n_samples = X.shape[0]
        obj_prim = 0.5 * norm(beta) ** 2 + C * np.sum(np.maximum(
            np.ones(n_samples) - (X @ beta) * y, np.zeros(n_samples)))
        obj_dual = 0.5 * beta.T @ beta - np.sum(r)
        return (obj_dual + obj_prim)

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask]

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is not None:
            dbeta0_new = init_dbeta0_new(jac0, mask, mask0)
        else:
            dbeta0_new = np.zeros(size_mat)
        return dbeta0_new

    def _init_dbeta(self, n_features):
        if self.dbeta is not None:
            return self.dbeta
        else:
            return np.zeros(n_features)

    def _init_dr(self, dbeta, X, y, sign_beta, C):
        is_sparse = issparse(X)
        sign = np.zeros(self.r.shape[0])
        sign[self.r == 0.0] = -1.0
        sign[self.r == C] = 1.0
        dr = np.zeros(X.shape[0])
        self.dr = dr
        if np.sum(sign == 1.0) != 0:
            dr[sign == 1.0] = np.repeat(C, (sign == 1).sum())
        if is_sparse:
            self.dbeta = np.array(np.sum(X.T.multiply(y * dr), axis=1))
        else:
            self.dbeta = np.sum(y * dr * X.T, axis=1)
        return dr

    @staticmethod
    @njit
    def _update_only_jac(Xs, ys, r, dbeta, dr, L, C, sign_beta):
        sign = np.zeros(r.shape[0])
        sign[r == 0.0] = -1.0
        sign[r == C] = 1.0
        for j in np.arange(0, Xs.shape[0])[sign == 0.0]:
            dF = ys[j] * np.sum(dbeta * Xs[j, :])
            dr_old = dr[j]
            dzj = dr[j] - (dF / L[j])
            dr[j] = dzj
            dbeta += (dr[j] - dr_old) * ys[j] * Xs[j, :]

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, C, sign_beta):
        sign = np.zeros(r.shape[0])
        sign[r == 0.0] = -1.0
        sign[r == C] = 1.0
        for j in np.arange(0, n_samples)[sign == 0.0]:
            # get the i-st row of X in sparse format
            Xis = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dF = y[j] * np.sum(dbeta[idx_nz] * Xis)
            dr_old = dr[j]
            dzj = dr[j] - (dF / L[j])
            dr[j] = dzj
            dbeta[idx_nz] += ((dr[j] - dr_old) * y[j] * Xis)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        if is_sparse:
            return slinalg.norm(X, axis=1) ** 2
        else:
            return norm(X, axis=1) ** 2

    @staticmethod
    def reduce_X(X, mask):
        return X

    @staticmethod
    def reduce_y(y, mask):
        return y

    def sign(self, x, log_C):
        sign = np.zeros(x.shape[0])
        sign[np.isclose(x, 0.0)] = -1.0
        sign[np.isclose(x, np.exp(log_C))] = 1.0
        return sign

    def get_dual_v(self, X, y, v, log_C):
        if issparse(X):
            v_dual = v * (X.T).multiply(y)
            v_dual = np.sum(v_dual, axis=1)
            v_dual = np.squeeze(np.array(v_dual))
        else:
            v_dual = (y * X.T).T @ v
        return v_dual

    @staticmethod
    def get_jac_v(X, y, mask, dense, jac, v):
        return jac[mask].T @ v(mask, dense)

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    def get_hessian(self, X, y, mask, dense, log_C):
        C = np.exp(log_C)
        full_supp = np.logical_and(self.r != 0, self.r != C)
        if issparse(X):
            Xy = X[full_supp, :].multiply(y[full_supp, np.newaxis])
            return Xy @ Xy.T
        else:
            Xy = (y[full_supp] * X[full_supp, :].T)
            return Xy.T @ Xy

    def _get_jac_t_v(self, X, y, jac, mask, dense, C, v, n_samples):
        C = C[0]
        full_supp = np.logical_and(self.r != 0, self.r != C)
        maskC = self.r == C
        hessian = (y[full_supp] * X[full_supp, :].T).T @ \
            (y[maskC] * X[maskC, :].T)
        hessian_vec = hessian @ np.repeat(C, maskC.sum())
        jac_t_v = hessian_vec.T @ jac
        jac_t_v += np.repeat(C, maskC.sum()).T @ v[maskC]
        return jac_t_v

    def restrict_full_supp(self, X, v, log_C):
        full_supp = np.logical_and(self.r != 0, self.r != np.exp(log_C))
        return v[full_supp]

    def proj_hyperparam(self, X, y, log_alpha):
        if log_alpha < -16.0:
            log_alpha = -16.0
        elif log_alpha > 4:
            log_alpha = 4
        return log_alpha

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta, dbeta, r, dr, C):
        maskC = r == C
        full_supp = np.logical_and(r != 0, r != C)
        dryX = dr[full_supp].T @ (ys[full_supp] * Xs[full_supp, :].T).T
        quadratic_term = dryX.T @ dryX
        if maskC.sum() != 0:
            linear_term = dryX.T @ (ys[maskC] * Xs[maskC, :].T) @ r[maskC]
        else:
            linear_term = 0
        res = quadratic_term + linear_term
        return norm(res)

    def _use_estimator(self, X, y, C, tol, max_iter):
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        self.estimator.set_params(
            tol=tol, C=C,
            fit_intercept=False, max_iter=max_iter)
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
        return mask, dense, None
