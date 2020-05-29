import numpy as np
from numpy.linalg import norm
from numba import njit
from sparse_ho.utils import ST, init_dbeta0_new, init_dbeta0_new_p
from sparse_ho.utils import proj_box_svm, compute_grad_proj, ind_box
from sparse_ho.utils import sigma
import scipy.sparse.linalg as slinalg


class Lasso():
    def __init__(
            self, X, y, log_alpha, log_alpha_max=None, max_iter=100, tol=1e-3):
        self.X = X
        self.y = y
        self.log_alpha = log_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.log_alpha_max = log_alpha_max

    def _init_dbeta_dr(self, X, mask0=None, jac0=None,
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
    def _init_dr(dbeta, X):
        return - X @ dbeta

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return 0.0
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, r, dbeta, dr, L, alpha, sign_beta):
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
            data, indptr, indices, n_samples, n_features,
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

    @staticmethod
    def _reduce_jac_t_v(jac, mask, dense, alphas):
        return alphas[mask] * np.sign(dense) @ jac

    def proj_param(self, log_alpha):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= self.X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        if log_alpha < -12:
            return - 12.0
        elif log_alpha > self.log_alpha_max + np.log(0.9):
            return self.log_alpha_max + np.log(0.9)
        else:
            return log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        print(is_sparse)
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    @staticmethod
    def hessian_f(x):
        return np.ones(np.size(x))


class wLasso():
    def __init__(self, X, y, log_alpha, log_alpha_max=None,
                 max_iter=100, tol=1e-3):
        self.X = X
        self.y = y
        self.log_alpha = log_alpha
        self.log_alpha_max = log_alpha_max
        self.max_iter = max_iter
        self.tol = tol

    def _init_dbeta_dr(self, X, mask0=None, jac0=None,
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
    def _init_dr(dbeta, X):
        return - X @ dbeta

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return np.zeros(self.log_alpha.shape[0])
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, r, dbeta, dr, L, alpha, sign_beta):
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
            data, indptr, indices, n_samples, n_features, dbeta, r, dr, L,
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

    @staticmethod
    def _reduce_jac_t_v(jac, mask, dense, alphas):
        size_supp = mask.sum()
        jac_t_v = np.zeros(size_supp)
        jac_t_v = alphas[mask] * np.sign(dense) * jac
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
            proj_log_alpha[proj_log_alpha > self.log_alpha_max] = self.log_alpha_max
        return proj_log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    @staticmethod
    def hessian_f(x):
        return np.ones(np.size(x))


class SVM():
    def __init__(self, logC, max_iter=100, tol=1e-3):
        self.logC = logC
        self.max_iter = max_iter
        self.tol = tol

    def _init_dbeta_dtheta(
            self, X, y, mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dtheta = np.zeros(n_samples)
        if jac0 is not None:
            dtheta = jac0.copy()
            dbeta = np.sum(y * dtheta * X.T, axis=1)
        else:
            dbeta = np.zeros(n_features)
        return dbeta, dtheta

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, theta, dtheta, C, Q, tol, compute_jac=True):

        n_samples = Q.shape[0]
        violator = 0.0
        for j in np.random.choice(n_samples, n_samples, replace=False):
            F = y[j] * np.sum(beta * X[j, :]) - 1.0
            grad_proj = compute_grad_proj(theta[j], F, C)
            if np.abs(grad_proj) > violator:
                violator = grad_proj
            if np.abs(grad_proj) > tol:
                theta_old = theta[j]
                zj = theta[j] - F / Q[j]
                theta[j] = proj_box_svm(zj, C)
                beta += (theta[j] - theta_old) * y[j] * X[j, :]
                if compute_jac:
                    dF = y[j] * np.sum(dbeta * X[j, :])
                    dtheta_old = dtheta[j]
                    dzj = dtheta[j] - dF / Q[j]
                    dtheta[j] = ind_box(zj, C) * dzj
                    dtheta[j] += C * (C <= zj)
                    dbeta += (dtheta[j] - dtheta_old) * y[j] * X[j, :]
# TODO
    # @staticmethod
    # @njit
    # def _update_beta_jac_bcd_sparse(
    #         data, indptr, indices, n_samples, n_features, beta,
    #         dbeta, r, dr, alphas, L, compute_jac=True):

    #     non_zeros = np.where(L != 0)[0]

    #     for j in non_zeros:
    #         # get the j-st column of X in sparse format
    #         Xjs = data[indptr[j]:indptr[j+1]]
    #         # get the non zero indices
    #         idx_nz = indices[indptr[j]:indptr[j+1]]
    #         beta_old = beta[j]
    #         if compute_jac:
    #             dbeta_old = dbeta[j]
    #         zj = beta[j] + r[idx_nz] @ Xjs / (L[j] * n_samples)
    #         beta[j:j+1] = ST(zj, alphas[j] / L[j])
    #         if compute_jac:
    #             dzj = dbeta[j] + Xjs @ dr[idx_nz] / (L[j] * n_samples)
    #             dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
    #             dbeta[j:j+1] -= alphas[j] * np.sign(beta[j]) / L[j]
    #             # update residuals
    #             dr[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)
    #         r[idx_nz] -= Xjs * (beta[j] - beta_old)

    @staticmethod
    def _get_pobj(X, y, beta, C):
        n_samples = X.shape[0]
        return (
            0.5 * norm(beta) ** 2 + C * np.sum(np.maximum(np.ones(n_samples) - y * (X @ beta), np.zeros(n_samples))))

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
    @njit
    def _update_only_jac(Xs, ys, C, dbeta, dtheta, Qs, alpha, thetas):
        n_samples = Xs.shape[0]
        for j in range(n_samples):
            dF = ys[j] * np.sum(dbeta * Xs[j, :])
            dtheta_old = dtheta[j]
            dzj = dtheta[j] - (dF / Qs[j])
            dtheta[j] = ind_box(thetas[j], C) * dzj
            dtheta[j] += C * (C == thetas[j])
            dbeta += (dtheta[j] - dtheta_old) * ys[j] * Xs[j, :]
# TODO
    # @staticmethod
    # @njit
    # def _update_only_jac_sparse(
    #         data, indptr, indices, n_samples, n_features,
    #         dbeta, dr, L, alpha, sign_beta):
    #     for j in range(n_features):
    #         # get the j-st column of X in sparse format
    #         Xjs = data[indptr[j]:indptr[j+1]]
    #         # get the non zero idices
    #         idx_nz = indices[indptr[j]:indptr[j+1]]
    #         # store old beta j for fast update
    #         dbeta_old = dbeta[j]
    #         # update of the Jacobian dbeta
    #         dbeta[j] += Xjs @ dr[idx_nz] / (L[j] * n_samples)
    #         dbeta[j] -= alpha * sign_beta[j] / L[j]
    #         dr[idx_nz] -= Xjs * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha[mask]

    @staticmethod
    def _reduce_jac_t_v(jac, mask, dense, alphas):
        return alphas[mask] * np.sign(dense) @ jac


class SparseLogreg():
    def __init__(
            self, X, y, log_alpha, log_alpha_max=None, max_iter=100, tol=1e-3):
        self.X = X
        self.y = y
        self.log_alpha = log_alpha
        self.max_iter = max_iter
        self.tol = tol
        self.log_alpha_max = log_alpha_max

    def _init_dbeta_dr(self, X, dense0=None,
                       mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros(n_features)
        if jac0 is None or not compute_jac:
            dr = np.zeros(n_samples)
        else:
            dbeta[mask0] = jac0.copy()
            dr = X[:, mask0] @ jac0.copy()
        return dbeta, dr

    def _init_beta_r(self, X, y, mask0, dense0):
        beta = np.zeros(X.shape[1])
        if dense0 is None:
            r = np.zeros(X.shape[0])
        else:
            beta[mask0] = dense0
            r = X[:, mask0] @ dense0
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
            grad_fj = sigma(r) - y
            zj = beta[j] - grad_fj @ X[:, j] / (L[j] * n_samples)
            beta[j] = ST(zj, alpha[j] / L[j])
            r += X[:, j] * (beta[j] - beta_old)
            if compute_jac:
                hess_fj = sigma(r) * (1 - sigma(r))
                dzj = dbeta[j] - X[:, j] @ (hess_fj * dr) / (L[j] * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alpha[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dr += X[:, j] * (dbeta[j] - dbeta_old)

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
            grad_fj = sigma(r) - y
            zj = beta[j] - grad_fj[idx_nz] @ Xjs / (L[j] * n_samples)
            beta[j:j+1] = ST(zj, alphas[j] / L[j])
            if compute_jac:
                hess_fj = sigma(r) * (1 - sigma(r))
                dzj = dbeta[j] - Xjs @ (hess_fj[idx_nz] * dr[idx_nz]) / (L[j] * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alphas[j] * np.sign(beta[j]) / L[j]
                # update residuals
                dr[idx_nz] += Xjs * (dbeta[j] - dbeta_old)
            r[idx_nz] += Xjs * (beta[j] - beta_old)

    @staticmethod
    @njit
    def _update_bcd_jac_backward(X, alpha, grad, beta, v_t_jac, L):
        sign_beta = np.sign(beta)
        r = X @ beta
        n_samples, n_features = X.shape
        for j in (np.arange(sign_beta.shape[0] - 1, -1, -1)):
            hess_fj = sigma(r) * (1 - sigma(r))
            grad -= (v_t_jac[j]) * alpha * sign_beta[j] / L[j]
            v_t_jac[j] *= np.abs(sign_beta[j])
            v_t_jac -= v_t_jac[j] / (L[j] * n_samples) * (X[:, j] * hess_fj) @ X
            r += X[:, j] * (beta[j-1] - beta[j])

        return grad

    @staticmethod
    def _get_pobj(r, beta, alphas, y):
        n_samples = r.shape[0]
        temp = sigma(r)

        return (
            np.sum(- y * np.log(temp) - (1 - y) * np.log(1 - temp)) / (n_samples) + np.abs(alphas * beta).sum())

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
    def _init_dr(dbeta, X):
        return X @ dbeta

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return 0.0
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, r, dbeta, dr, L, alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            dbeta_old = dbeta[j]
            hess_fj = sigma(r) * (1 - sigma(r))
            dbeta[j:j+1] += - Xs[:, j] @ (hess_fj * dr) / (L[j] * n_samples)
            dbeta[j:j+1] -= alpha * sign_beta[j] / L[j]
            # update residuals
            dr += Xs[:, j] * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, n_samples, n_features,
            dbeta, r, dr, L, alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dbeta_old = dbeta[j]
            hess_fj = sigma(r) * (1 - sigma(r))
            # update of the Jacobian dbeta
            dbeta[j] -= Xjs @ (hess_fj[idx_nz] * dr[idx_nz]) / (L[j] * n_samples)
            dbeta[j] -= alpha * sign_beta[j] / L[j]
            dr[idx_nz] += Xjs * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    @staticmethod
    def _reduce_jac_t_v(jac, mask, dense, alphas):
        return alphas[mask] * np.sign(dense) @ jac

    def proj_param(self, log_alpha):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= (4 * self.X.shape[0])
            self.log_alpha_max = np.log(alpha_max)
        if log_alpha < -12:
            return - 12.0
        elif log_alpha > self.log_alpha_max + np.log(0.9):
            return self.log_alpha_max + np.log(0.9)
        else:
            return log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (4 * X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (4 * X.shape[0])

    @staticmethod
    def hessian_f(x):
        return sigma(x) * (1 - sigma(x))
