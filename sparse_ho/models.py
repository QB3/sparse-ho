import numpy as np
from sklearn import linear_model
from numpy.linalg import norm
from numba import njit
from sparse_ho.utils import ST, init_dbeta0_new, init_dbeta0_new_p
from sparse_ho.utils import proj_box_svm, ind_box, compute_grad_proj
from sparse_ho.utils import sigma
import scipy.sparse.linalg as slinalg
from scipy.sparse import issparse, csc_matrix


class Lasso():
    def __init__(
            self, X, y, log_alpha, log_alpha_max=None, max_iter=100, tol=1e-3, use_sk=False):
        self.X = X
        self.y = y
        self.log_alpha = log_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.log_alpha_max = log_alpha_max
        if use_sk:
            self.clf = linear_model.Lasso(
                fit_intercept=False, max_iter=max_iter, warm_start=True,
                solver='liblinear')
        else:
            self.clf = None

    def _init_dbeta_dr(self, X, y, mask0=None, jac0=None,
                       dense0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros(n_features)
        if jac0 is None or not compute_jac:
            dr = np.zeros(n_samples)
        else:
            dbeta[mask0] = jac0.copy()
            dr = - X[:, mask0] @ jac0.copy()
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0=None, dense0=None):
        beta = np.zeros(X.shape[1])
        if dense0 is None or len(dense0) == 0:
            r = y.copy()
            r = r.astype(np.float)
        else:
            beta[mask0] = dense0.copy()
            r = y - X[:, mask0] @ dense0
        return beta, r

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, alpha, L, compute_jac=True):
        n_samples, n_features = X.shape
        non_zeros = np.where(L != 0)[0]

        for j in non_zeros:
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
                # compute derivatives
            zj = beta[j] + r @ X[:, j] / (L[j] * n_samples)
            beta[j] = ST(zj, alpha[j] / L[j])
            # beta[j:j+1] = ST(zj, alpha[j] / L[j])
            if compute_jac:
                dzj = dbeta[j] + X[:, j] @ dr / (L[j] * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alpha[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dr -= X[:, j] * (dbeta[j] - dbeta_old)
            r -= X[:, j] * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, r, dr, alphas, L, compute_jac=True):

        non_zeros = np.where(L != 0)[0]

        for j in non_zeros:
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero indices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
            zj = beta[j] + r[idx_nz] @ Xjs / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alphas[j] / L[j])
            if compute_jac:
                dzj = dbeta[j] + Xjs @ dr[idx_nz] / (L[j] * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alphas[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dr[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)
            r[idx_nz] -= Xjs * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_bcd_jac_backward(X, alpha, grad, beta, v_t_jac, L):
        sign_beta = np.sign(beta)
        n_samples, n_features = X.shape
        for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
            grad -= (v_t_jac[j]) * alpha * sign_beta[j] / L[j]
            v_t_jac[j] *= np.abs(sign_beta[j])
            v_t_jac -= v_t_jac[j] / (L[j] * n_samples) * X[:, j] @ X

        return grad

    @staticmethod
    def _get_pobj0(r, beta, alphas, y=None):
        n_samples = r.shape[0]
        return norm(y) ** 2 / (2 * n_samples)

    @staticmethod
    def _get_pobj(r, beta, alphas, y=None):
        n_samples = r.shape[0]
        return (
            norm(r) ** 2 / (2 * n_samples) + np.abs(alphas * beta).sum())

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask]

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        return jac_v

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is not None:
            dbeta0_new = init_dbeta0_new(jac0, mask, mask0)
        else:
            dbeta0_new = np.zeros(size_mat)
        return dbeta0_new

    @staticmethod
    def _init_dbeta(n_features):
        dbeta = np.zeros(n_features)
        return dbeta

    @staticmethod
    def _init_dr(dbeta, X, y):
        return - X @ dbeta

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return 0.0
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, r, dbeta, dr, L, alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            # dbeta_old = dbeta[j].copy()
            dbeta_old = dbeta[j]
            dbeta[j] += Xs[:, j].T @ dr / (L[j] * n_samples)
            dbeta[j] -= alpha * sign_beta[j] / L[j]
            dr -= Xs[:, j] * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dbeta_old = dbeta[j]
            # update of the Jacobian dbeta
            dbeta[j] += Xjs @ dr[idx_nz] / (L[j] * n_samples)
            dbeta[j] -= alpha * sign_beta[j] / L[j]
            dr[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    # @staticmethod
    def _get_jac_t_v(self, jac, mask, dense, alphas, v):
        n_samples = self.X.shape[0]
        return n_samples * alphas[mask] * np.sign(dense) @ jac

    def proj_param(self, log_alpha):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= self.X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        if log_alpha < self.log_alpha_max - 7:
            return self.log_alpha_max - 7
        elif log_alpha > self.log_alpha_max + np.log(0.9):
            return self.log_alpha_max + np.log(0.9)
        else:
            return log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        # print(is_sparse)
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    def sk(self, X, y, alpha, tol, max_iter):
        if self.clf is None:
            self.clf = linear_model.Lasso(
                fit_intercept=False, max_iter=max_iter, warm_start=True)
        self.clf.alpha = alpha
        self.clf.tol = tol
        # clf = linear_model.Lasso(
        #     alpha=alpha, fit_intercept=False, tol=tol, max_iter=max_iter)
        self.clf.fit(X, y)
        mask = self.clf.coef_ != 0
        dense = self.clf.coef_[mask]
        return mask, dense, None

    def reduce_X(self, mask):
        return self.X[:, mask]

    def reduce_y(self, mask):
        return self.y

    def sign(self, x):
        return np.sign(x)

    def get_primal(self, mask, dense):
        return mask, dense

    def get_jac_v(self, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    def get_hessian(self, mask, dense):
        hessian = self.X[:, mask].T @ self.X[:, mask]
        return hessian

    def restrict_full_supp(self, mask, dense, v):
        return v

    def compute_alpha_max(self):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= self.X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        return self.log_alpha_max

    def get_jac_obj(self, Xs, ys, sign_beta, dbeta, r, dr, alpha):
        n_samples = self.X.shape[0]
        return(
            norm(dr.T @ dr + n_samples * alpha * sign_beta @ dbeta))


class wLasso():
    def __init__(self, X, y, log_alpha, log_alpha_max=None,
                 max_iter=100, tol=1e-3):
        self.X = X
        self.y = y
        self.log_alpha = log_alpha
        self.log_alpha_max = log_alpha_max
        self.max_iter = max_iter
        self.tol = tol

    def _init_dbeta_dr(self, X, y, mask0=None, jac0=None,
                       dense0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros((n_features, n_features))
        dr = np.zeros((n_samples, n_features))
        if jac0 is not None:
            dbeta[np.ix_(mask0, mask0)] = jac0.copy()
            dr[:, mask0] = - X[:, mask0] @ jac0
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0=None, dense0=None):
        beta = np.zeros(X.shape[1])
        if dense0 is None or len(dense0) == 0:
            r = y.copy()
            r = r.astype(np.float)
        else:
            beta[mask0] = dense0.copy()
            r = y - X[:, mask0] @ dense0
        return beta, r

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, alpha, L, compute_jac=True):
        n_samples, n_features = X.shape
        non_zeros = np.where(L != 0)[0]

        for j in non_zeros:
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j, :].copy()
            zj = beta[j] + r @ X[:, j] / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alpha[j] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + X[:, j] @ dr / (L[j] * n_samples)
                dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, j] -= alpha[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dr -= np.outer(X[:, j], (dbeta[j, :] - dbeta_old))
            r -= X[:, j] * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, r, dr, alphas, L, compute_jac=True):
        non_zeros = np.where(L != 0)[0]

        for j in non_zeros:
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            ###########################################
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j, :].copy()
            zj = beta[j] + r[idx_nz] @ Xjs / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alphas[j] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + Xjs @ dr[idx_nz, :] / (L[j] * n_samples)
                dbeta[j:j+1, :] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, j] -= alphas[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dr[idx_nz, :] -= np.outer(Xjs, (dbeta[j, :] - dbeta_old))
            r[idx_nz] -= Xjs * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_bcd_jac_backward(
            X, alpha, jac_t_v, beta, v_, L):
        n_samples, n_features = X.shape
        sign_beta = np.sign(beta)
        for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
            jac_t_v[j] = jac_t_v[j] - (v_[j]) * alpha[j] * sign_beta[j] / L[j]
            v_[j] *= np.abs(sign_beta[j])
            v_ -= v_[j] / (L[j] * n_samples) * X[:, j] @ X
        return jac_t_v

    @staticmethod
    def _get_pobj(r, beta, alphas, y=None):
        n_samples = r.shape[0]
        return (
            norm(r) ** 2 / (2 * n_samples) + norm(alphas * beta, 1))

    @staticmethod
    def _get_pobj0(r, beta, alphas, y=None):
        n_samples = r.shape[0]
        return norm(y) ** 2 / (2 * n_samples)

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[np.ix_(mask, mask)]

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is None:
            dbeta0_new = np.zeros((size_mat, size_mat))
        else:
            dbeta0_new = init_dbeta0_new_p(jac0, mask, mask0)
        return dbeta0_new

    @staticmethod
    def _init_dbeta(n_features):
        dbeta = np.zeros((n_features, n_features))
        return dbeta

    @staticmethod
    def _init_dr(dbeta, X, y):
        return - X @ dbeta

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return np.zeros(self.log_alpha.shape[0])
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, r, dbeta, dr, L, alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            dbeta_old = dbeta[j, :].copy()
            dbeta[j:j+1, :] = dbeta[j, :] + Xs[:, j] @ dr / (L[j] * n_samples)
            dbeta[j:j+1, j] -= alpha[j] * sign_beta[j] / L[j]
            # update residuals
            dr -= np.outer(Xs[:, j], (dbeta[j, :] - dbeta_old))

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features, dbeta, r, dr, L,
            alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dbeta_old = dbeta[j, :].copy()

            dbeta[j:j+1, :] += Xjs @ dr[idx_nz] / (L[j] * n_samples)
            dbeta[j, j] -= alpha[j] * sign_beta[j] / L[j]
            dr[idx_nz] -= np.outer(Xjs, (dbeta[j] - dbeta_old))

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha[mask]

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        res = np.zeros(n_features)
        res[mask] = jac_v
        return res

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        return jac_v[mask]

    # @staticmethod
    def _get_jac_t_v(self, jac, mask, dense, alphas, v):
        n_samples = self.X.shape[0]
        size_supp = mask.sum()
        jac_t_v = np.zeros(size_supp)
        jac_t_v = n_samples * alphas[mask] * np.sign(dense) * jac
        return jac_t_v

    def proj_param(self, log_alpha):
        """Maybe we could do this in place.
        """
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= self.X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        proj_log_alpha = log_alpha.copy()
        proj_log_alpha[proj_log_alpha < -12] = -12
        if np.max(proj_log_alpha > self.log_alpha_max):
            proj_log_alpha[
                proj_log_alpha > self.log_alpha_max] = self.log_alpha_max
        return proj_log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    def get_hessian(self, mask, dense):
        hessian = self.X[:, mask].T @ self.X[:, mask]
        return hessian

    def sk(self, X, y, alpha, tol, max_iter):
        """TODO
        """
        X /= alpha
        clf = linear_model.Lasso(
            alpha=1, fit_intercept=False, tol=tol, max_iter=max_iter)
        clf.fit(X, y)
        mask = clf.coef_ != 0
        dense = (clf.coef_ / alpha)[mask]
        return mask, dense, None

    def reduce_X(self, mask):
        return self.X[:, mask]

    def reduce_y(self, mask):
        return self.y

    def sign(self, x):
        return np.sign(x)

    def get_primal(self, mask, dense):
        return mask, dense

    def get_jac_v(self, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    def restrict_full_supp(self, mask, dense, v):
        return v

    def get_jac_obj(self, Xs, ys, sign_beta, dbeta, r, dr, alpha):
        n_samples = self.X.shape[0]
        return(
            norm(dr.T @ dr + n_samples * alpha * sign_beta @ dbeta))


class SVM():
    def __init__(self, X, y, logC, max_iter=100, tol=1e-3):
        self.logC = logC
        self.max_iter = max_iter
        self.tol = tol
        self.X = X
        self.y = y

    def _init_dbeta_dr(self, X, y, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros(n_samples)
        if jac0 is None or not compute_jac:
            dr = np.zeros(n_features)
        else:
            dbeta[mask0] = jac0.copy()
        if issparse(self.X):
            dr = (self.X.T).multiply(y * dbeta)
            dr = np.sum(dr, axis=1)
            dr = np.squeeze(np.array(dr))
        else:
            dr = np.sum(y * dbeta * X.T, axis=1)
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0, dense0):
        beta = np.zeros(X.shape[0])
        if dense0 is None:
            r = np.zeros(X.shape[1])
        else:
            beta[mask0] = dense0
            if issparse(self.X):
                r = np.sum(self.X.T.multiply(y * beta), axis=1)
                r = np.squeeze(np.array(r))
            else:
                r = np.sum(y * beta * X.T, axis=1)
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
            F = y[j] * np.sum(r * X[j, :]) - 1.0
            beta_old = beta[j]
            zj = beta[j] - F / L[j]
            beta[j] = proj_box_svm(zj, C)
            r += (beta[j] - beta_old) * y[j] * X[j, :]
            if compute_jac:
                dF = y[j] * np.sum(dr * X[j, :])
                dbeta_old = dbeta[j]
                dzj = dbeta[j] - dF / L[j]
                dbeta[j] = ind_box(zj, C) * dzj
                dbeta[j] += C * (C <= zj)
                dr += (dbeta[j] - dbeta_old) * y[j] * X[j, :]

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
            G = y[j] * np.sum(r[idx_nz] * Xis) - 1.0

            # compute projected gradient
            PG = compute_grad_proj(beta[j], G, C)

            if np.abs(PG) > 1e-12:
                beta_old = beta[j]
                # update one coefficient SVM
                zj = beta[j] - G / L[j]
                beta[j] = min(max(zj, 0), C)
                r[idx_nz] += (beta[j] - beta_old) * y[j] * Xis
                if compute_jac:
                    dbeta_old = dbeta[j]
                    dG = y[j] * np.sum(dr[idx_nz] * Xis)
                    dzj = dbeta[j] - dG / L[j]
                    dbeta[j:j+1] = ind_box(zj, C) * dzj
                    dbeta[j:j+1] += C * (C <= zj)
                    # update residuals
                    dr[idx_nz] += (dbeta[j] - dbeta_old) * y[j] * Xis

        # import ipdb; ipdb.set_trace()

    def _get_pobj0(self, r, beta, C, y):
        C = C[0]
        n_samples = self.X.shape[0]
        obj_prim = C * np.sum(np.maximum(
            np.ones(n_samples), np.zeros(n_samples)))
        return obj_prim

    def _get_pobj(self, r, beta, C, y):
        # r = y.copy()
        C = C[0]
        n_samples = self.X.shape[0]
        obj_prim = 0.5 * norm(r) ** 2 + C * np.sum(np.maximum(
            np.ones(n_samples) - (self.X @ r) * self.y, np.zeros(n_samples)))
        obj_dual = 0.5 * r.T @ r - np.sum(beta)
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

    @staticmethod
    def _init_dbeta(n_features):
        dbeta = np.zeros(n_features)
        return dbeta

    @staticmethod
    def _init_dr(dbeta, X, y):
        is_sparse = issparse(X)
        if is_sparse:
            res = np.array(np.sum(X.T.multiply(y * dbeta), axis=1))
            return res.reshape((res.shape[0],))
        else:
            return np.sum(y * dbeta * X.T, axis=1)

    @staticmethod
    @njit
    def _update_only_jac(Xs, ys, r, dbeta, dr, L, C, sign_beta):
        supp = np.where(sign_beta == 0.0)
        dbeta[sign_beta == 1.0] = C
        dr = np.sum(ys * dbeta * Xs.T, axis=1)
        for j in supp[0]:
            dF = ys[j] * np.sum(dr * Xs[j, :])
            dbeta_old = dbeta[j]
            dzj = dbeta[j] - (dF / L[j])
            dbeta[j] = dzj
            dr += (dbeta[j] - dbeta_old) * ys[j] * Xs[j, :]

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, C, sign_beta):
        supp = np.where(sign_beta == 0.0)
        for j in np.where(sign_beta == 1.0)[0]:
            Xis = data[indptr[j]:indptr[j+1]]
            idx_nz = indices[indptr[j]:indptr[j+1]]
            dr[idx_nz] += ((C - dbeta[j]) * y[j] * Xis)
        dbeta[sign_beta == 1.0] = C
        for j in supp[0]:
            # get the i-st row of X in sparse format
            Xis = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dF = y[j] * np.sum(dr[idx_nz] * Xis)
            dbeta_old = dbeta[j]
            dzj = dbeta[j] - (dF / L[j])
            dbeta[j] = dzj
            dr[idx_nz] += ((dbeta[j] - dbeta_old) * y[j] * Xis)

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

    def reduce_X(self, mask):
        return self.X[mask, :]

    def reduce_y(self, mask):
        return self.y[mask]

    def sign(self, x):
        sign = np.zeros(x.shape[0])
        sign[np.isclose(x, 0.0)] = -1.0
        sign[np.isclose(x, np.exp(self.logC))] = 1.0
        return sign

    def get_jac_v(self, mask, dense, jac, v):
        n_samples, n_features = self.X.shape
        if issparse(self.X):
            primal_jac = np.sum(self.X[mask, :].T.multiply(self.y[mask] * jac), axis=1)
            primal_jac = np.squeeze(np.array(primal_jac))
            primal = np.sum(self.X[mask, :].T.multiply(self.y[mask] * dense), axis=1)
            primal = np.squeeze(np.array(primal))
        else:
            primal_jac = np.sum(self.y[mask] * jac * self.X[mask, :].T, axis=1)
            primal = np.sum(self.y[mask] * dense * self.X[mask, :].T, axis=1)
        mask_primal = np.repeat(True, primal.shape[0])
        dense_primal = primal[mask_primal]
        return primal_jac[primal_jac != 0].T @ v(mask_primal, dense_primal)[primal_jac != 0]

    def get_primal(self, mask, dense):
        if issparse(self.X):
            primal = np.sum(self.X[mask, :].T.multiply(self.y[mask] * dense), axis=1)
            primal = np.squeeze(np.array(primal))
        else:
            primal = np.sum(self.y[mask] * dense * self.X[mask, :].T, axis=1)
        mask_primal = primal != 0
        dense_primal = primal[mask_primal]
        return mask_primal, dense_primal

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    def get_hessian(self, mask, dense):
        beta = np.zeros(self.X.shape[0])
        beta[mask] = dense
        full_supp = np.logical_and(np.logical_not(np.isclose(beta, 0)), np.logical_not(np.isclose(beta, np.exp(self.logC))))

        if issparse(self.X):
            mat = self.X[full_supp, :].multiply(self.y[full_supp, np.newaxis])
        else:
            mat = self.y[full_supp, np.newaxis] * self.X[full_supp, :]
        Q = mat @ mat.T
        return Q

    def _get_jac_t_v(self, jac, mask, dense, C, v):
        C = C[0]
        n_samples = self.X.shape[0]
        beta = np.zeros(n_samples)
        beta[mask] = dense
        maskC = np.isclose(beta, C)
        full_supp = np.logical_and(np.logical_not(np.isclose(beta, 0)), np.logical_not(np.isclose(beta, C)))

        full_jac = np.zeros(n_samples)
        full_jac[full_supp] = jac
        full_jac[maskC] = C

        # primal dual relation
        jac_primal = (self.y[mask] * full_jac[mask]) @ self.X[mask, :]

        return jac_primal @ v

        # if issparse(self.X):
        #     mat = self.X[full_supp, :].multiply(self.y[full_supp, np.newaxis])
        #     Q = mat @ (self.X[maskC, :].multiply(self.y[maskC, np.newaxis])).T
        # else:
        #     mat = self.y[full_supp, np.newaxis] * self.X[full_supp, :]
        #     Q = mat @ (self.y[maskC, np.newaxis] * self.X[maskC, :]).T

        # u = (np.eye(Q.shape[0], Q.shape[1]) - Q) @ (np.ones(maskC.sum()) * C)
        # if issparse(self.X):
        #     temp = self.X[maskC, :].multiply(self.y[maskC, np.newaxis])
        #     w = temp @ v
        # else:
        #     w = ((self.y[maskC, np.newaxis] * self.X[maskC, :]) @ v)

        # if issparse(self.X):
        #     return np.array(u @ jac + C * np.sum(w))[0]
        # else:
        #     return np.array(u @ jac + C * np.sum(w))

    def restrict_full_supp(self, mask, dense, v):
        C = np.exp(self.logC)
        n_samples = self.X.shape[0]
        beta = np.zeros(n_samples)
        beta[mask] = dense
        maskC = np.isclose(beta, C)
        full_supp = np.logical_and(np.logical_not(np.isclose(beta, 0)), np.logical_not(np.isclose(beta, C)))
        if issparse(self.X):
            mat = self.X[full_supp, :].multiply(self.y[full_supp, np.newaxis])
            Q = mat @ (self.X[maskC, :].multiply(self.y[maskC, np.newaxis])).T
        else:
            mat = self.y[full_supp, np.newaxis] * self.X[full_supp, :]
            Q = mat @ (self.y[maskC, np.newaxis] * self.X[maskC, :]).T

        w = (np.eye(Q.shape[0], Q.shape[1]) - Q) @ (np.ones(maskC.sum()) * C)
        if issparse(self.X):
            return - np.array(w)[0]
        else:
            return - w
        # n_samples = self.X.shape[0]
        # beta = np.zeros(n_samples)
        # beta[mask] = dense
        # full_supp = np.logical_and(np.logical_not(np.isclose(beta, 0)), np.logical_not(np.isclose(beta, np.exp(self.logC))))
        # if issparse(self.X):
        #     temp = self.X[full_supp, :].multiply(self.y[full_supp, np.newaxis])
        #     res = (temp @ v)
        # else:
        #     res = ((self.y[full_supp, np.newaxis] * self.X[full_supp, :]) @ v)
        # return - res

    def proj_param(self, log_alpha):
        if log_alpha < -16.0:
            log_alpha = -16.0
        elif log_alpha > 4:
            log_alpha = 4
        return log_alpha

    def get_jac_obj(self, Xs, ys, sign_beta, dbeta, r, dr, C):
        full_supp = sign_beta == 0.0
        maskC = sign_beta == 1.0
        if issparse(Xs):
            yXdbeta = (Xs[full_supp, :].multiply(ys[full_supp, np.newaxis])).T @ dbeta[full_supp]
        else:
            yXdbeta = (ys[full_supp, np.newaxis] * Xs[full_supp, :]).T @ dbeta[full_supp]
        q = yXdbeta.T @ yXdbeta
        if issparse(Xs):
            linear_term = yXdbeta.T @ ((Xs[maskC, :].multiply(ys[maskC, np.newaxis])).T @ (np.ones(maskC.sum()) * C))
        else:
            linear_term = yXdbeta.T @ ((ys[maskC, np.newaxis] * Xs[maskC, :]).T @ (np.ones(maskC.sum()) * C))
        res = q + linear_term - C * np.sum(dbeta[full_supp])
        return(
            norm(res))


class SparseLogreg():
    def __init__(
            self, X, y, log_alpha, log_alpha_max=None, max_iter=100, tol=1e-3, use_sk=False, verbose=False):
        self.X = X
        self.y = y
        self.log_alpha = log_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.log_alpha_max = log_alpha_max
        self.verbose = verbose

        if use_sk:
            self.clf = linear_model.LogisticRegression(
                fit_intercept=False, max_iter=max_iter, warm_start=True,
                penalty='l1', verbose=self.verbose)
        else:
            self.clf = None

    def _init_dbeta_dr(self, X, y, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros(n_features)
        if jac0 is None or not compute_jac:
            dr = np.zeros(n_samples)
        else:
            dbeta[mask0] = jac0.copy()
            dr = y * (X[:, mask0] @ jac0.copy())
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0, dense0):
        beta = np.zeros(X.shape[1])
        if dense0 is None:
            r = np.zeros(X.shape[0])
        else:
            beta[mask0] = dense0
            r = y * (X[:, mask0] @ dense0)
        return beta, r

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, alpha, L, compute_jac=True):
        n_samples, n_features = X.shape
        for j in range(n_features):
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
                # compute derivatives
            sigmar = sigma(r)
            grad_j = X[:, j] @ (y * (sigmar - 1))
            L_temp = np.sum(X[:, j] ** 2 * sigmar * (1 - sigmar))
            L_temp /= n_samples
            zj = beta[j] - grad_j / (L_temp * n_samples)
            beta[j] = ST(zj, alpha[j] / L_temp)
            r += y * X[:, j] * (beta[j] - beta_old)
            if compute_jac:
                dsigmar = sigmar * (1 - sigmar) * dr
                hess_fj = X[:, j] @ (y * dsigmar)
                dzj = dbeta[j] - hess_fj / (L_temp * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alpha[j] * np.sign(beta[j]) / L_temp
                # update residuals
                dr += y * X[:, j] * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, r, dr, alphas, L, compute_jac=True):

        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero indices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
            sigmar = sigma(r[idx_nz])
            grad_j = Xjs @ (y[idx_nz] * (sigmar - 1))
            L_temp = (Xjs ** 2 * sigmar * (1 - sigmar)).sum()
            # Xjs2 = (Xjs ** 2 * sigmar * (1 - sigmar)).sum()
            # temp1 =
            # # temp2 = temp1 * Xjs2
            # L_temp = temp2.sum()
            L_temp /= n_samples
            if L_temp != 0:
                zj = beta[j] - grad_j / (L_temp * n_samples)
                beta[j:j+1] = ST(zj, alphas[j] / L_temp)
                if compute_jac:
                    dsigmar = sigmar * (1 - sigmar) * dr[idx_nz]
                    hess_fj = Xjs @ (y[idx_nz] * dsigmar)
                    dzj = dbeta[j] - hess_fj / (L_temp * n_samples)
                    dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                    dbeta[j:j+1] -= alphas[j] * np.sign(beta[j]) / L_temp
                    # update residuals
                    dr[idx_nz] += y[idx_nz] * Xjs * (dbeta[j] - dbeta_old)
                r[idx_nz] += y[idx_nz] * Xjs * (beta[j] - beta_old)

    @staticmethod
    @njit
    # TODO
    def _update_bcd_jac_backward(X, alpha, grad, beta, v_t_jac, L):
        sign_beta = np.sign(beta)
        r = X @ beta
        n_samples, n_features = X.shape
        for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
            hess_fj = sigma(r) * (1 - sigma(r))
            grad -= (v_t_jac[j]) * alpha * sign_beta[j] / L[j]
            v_t_jac[j] *= np.abs(sign_beta[j])
            v_t_jac -= v_t_jac[j] / (
                L[j] * n_samples) * (X[:, j] * hess_fj) @ X
            r += X[:, j] * (beta[j-1] - beta[j])

        return grad

    @staticmethod
    def _get_pobj(r, beta, alphas, y):
        n_samples = r.shape[0]
        return (
            np.sum(np.log(1 + np.exp(- r))) / (n_samples) + np.abs(alphas * beta).sum())

    @staticmethod
    def _get_pobj0(r, beta, alphas, y):
        n_samples = r.shape[0]
        return np.log(2) / n_samples
        # return (np.sum(np.log(1)) / (n_samples))
        # return (np.sum(np.log(1)) / (n_samples))

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask]

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        return jac_v

    @staticmethod
    def get_mask_jac_v(mask, jac_v):
        return jac_v

    @staticmethod
    def _init_dbeta0(mask, mask0, jac0):
        size_mat = mask.sum()
        if jac0 is not None:
            dbeta0_new = init_dbeta0_new(jac0, mask, mask0)
        else:
            dbeta0_new = np.zeros(size_mat)
        return dbeta0_new

    @staticmethod
    def _init_dbeta(n_features):
        dbeta = np.zeros(n_features)
        return dbeta

    @staticmethod
    def _init_dr(dbeta, X, y):
        return y * (X @ dbeta)

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return 0.0
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, r, dbeta, dr, L, alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            sigmar = sigma(r)
            L_temp = np.sum(Xs[:, j] ** 2 * sigmar * (1 - sigmar))
            L_temp /= n_samples

            dbeta_old = dbeta[j]
            dsigmar = sigmar * (1 - sigmar) * dr
            hess_fj = Xs[:, j] @ (y * dsigmar)
            dbeta[j:j+1] += - hess_fj / (L_temp * n_samples)
            dbeta[j:j+1] -= alpha * sign_beta[j] / L_temp
            # update residuals
            dr += y * Xs[:, j] * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            sigmar = sigma(r[idx_nz])
            L_temp = np.sum(Xjs ** 2 * sigmar * (1 - sigmar))
            L_temp /= n_samples
            if L_temp != 0:
                # store old beta j for fast update
                dbeta_old = dbeta[j]
                dsigmar = sigmar * (1 - sigmar) * dr[idx_nz]

                hess_fj = Xjs @ (y[idx_nz] * dsigmar)
                # update of the Jacobian dbeta
                dbeta[j] -= hess_fj / (L_temp * n_samples)
                dbeta[j] -= alpha * sign_beta[j] / L_temp
                dr[idx_nz] += y[idx_nz] * Xjs * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    # @staticmethod
    def _get_jac_t_v(self, jac, mask, dense, alphas, v):
        n_samples = self.X.shape[0]
        return n_samples * alphas[mask] * np.sign(dense) @ jac

    def proj_param(self, log_alpha):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= (4 * self.X.shape[0])
            self.log_alpha_max = np.log(alpha_max)
        if log_alpha < -18:
            return - 18.0
        elif log_alpha > self.log_alpha_max + np.log(0.9):
            return self.log_alpha_max + np.log(0.9)
        else:
            return log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        return 0.0

    def reduce_X(self, mask):
        return self.X[:, mask]

    def reduce_y(self, mask):
        return self.y

    def sign(self, x):
        return np.sign(x)

    def get_primal(self, mask, dense):
        return mask, dense

    def get_jac_v(self, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    def get_hessian(self, mask, dense):
        a = self.y * (self.X[:, mask] @ dense)
        temp = sigma(a) * (1 - sigma(a))
        is_sparse = issparse(self.X)
        if is_sparse:
            hessian = csc_matrix(
                self.X[:, mask].T.multiply(temp)) @ self.X[:, mask]
        else:
            hessian = (self.X[:, mask].T * temp) @ self.X[:, mask]
        return hessian

    def restrict_full_supp(self, mask, dense, v):
        return v

    def compute_alpha_max(self):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= (4 * self.X.shape[0])
            self.log_alpha_max = np.log(alpha_max)
        return self.log_alpha_max

    def get_jac_obj(self, Xs, ys, sign_beta, dbeta, r, dr, alpha):
        n_samples = self.X.shape[0]
        return(
            norm(dr.T @ dr + n_samples * alpha * sign_beta @ dbeta))

    def sk(self, X, y, alpha, tol, max_iter):
        n_samples = X.shape[0]
        if self.clf is None:
            self.clf = linear_model.LogisticRegression(
                fit_intercept=False, max_iter=max_iter, warm_start=True,
                penalty='l1', solver='liblinear', verbose=self.verbose)

        self.clf.C = 1 / (alpha * n_samples)
        self.clf.tol = tol
        self.clf.max_iter = max_iter
        # clf = linear_model.Lasso(
        #     alpha=alpha, fit_intercept=False, tol=tol, max_iter=max_iter)
        self.clf.fit(X, y)
        mask = self.clf.coef_ != 0
        dense = self.clf.coef_[mask]
        # import ipdb; ipdb.set_trace()
        return mask[0], dense, None
