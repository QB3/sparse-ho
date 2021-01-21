import numpy as np
from numpy.linalg import norm
import scipy.sparse.linalg as slinalg

from numba import njit

from sparse_ho.models.base import BaseModel
from sparse_ho.utils import prox_elasticnet, ST


class ElasticNet(BaseModel):
    def __init__(
            self, max_iter=1000, estimator=None, log_alpha_max=None):
        self.max_iter = max_iter
        self.log_alpha_max = log_alpha_max
        self.estimator = estimator

    def _init_dbeta_dr(self, X, y, mask0=None, jac0=None,
                       dense0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros((n_features, 2))
        if jac0 is None or not compute_jac:
            dr = np.zeros((n_samples, 2))
        else:
            dbeta[mask0, :] = jac0.copy()
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
                dbeta_old = dbeta[j, :].copy()
                # compute derivatives
            zj = beta[j] + r @ X[:, j] / (L[j] * n_samples)
            beta[j] = prox_elasticnet(zj, alpha[0] / L[j], alpha[1] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + X[:, j] @ dr / (L[j] * n_samples)
                dbeta[j:j+1, :] = (1 / (1 + alpha[1] / L[j])) * \
                    np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, 0] -= (alpha[0] * np.sign(beta[j])
                                    ) / L[j] / (1 + alpha[1] / L[j])
                dbeta[j:j+1, 1] -= (alpha[1] / L[j] * beta[j]
                                    ) / (1 + alpha[1] / L[j])
                # update residuals
                dr[:, 0] -= X[:, j] * (dbeta[j, 0] - dbeta_old[0])
                dr[:, 1] -= X[:, j] * (dbeta[j, 1] - dbeta_old[1])
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
                dbeta_old = dbeta[j, :].copy()
            zj = beta[j] + r[idx_nz] @ Xjs / (L[j] * n_samples)
            beta[j:j+1] = prox_elasticnet(zj,
                                          alphas[0] / L[j], alphas[1] / L[j])
            if compute_jac:
                dzj = dbeta[j, :] + Xjs @ dr[idx_nz, :] / (L[j] * n_samples)
                dbeta[j:j+1, :] = (1 / (1 + alphas[1] / L[j])) * \
                    np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1, 0] -= alphas[0] * \
                    np.sign(beta[j]) / L[j] / (1 + (alphas[1] / L[j]))
                dbeta[j:j+1, 1] -= (alphas[1] / L[j] * beta[j]
                                    ) / (1 + (alphas[1] / L[j]))
                # update residuals
                dr[idx_nz, 0] -= Xjs * (dbeta[j, 0] - dbeta_old[0])
                dr[idx_nz, 1] -= Xjs * (dbeta[j, 1] - dbeta_old[1])
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
    def _get_pobj(r, X, beta, alphas, y=None):
        n_samples = r.shape[0]
        pobj = norm(r) ** 2 / (2 * n_samples) + np.abs(alphas[0] * beta).sum()
        pobj += 0.5 * alphas[1] * norm(beta) ** 2
        return pobj

    @staticmethod
    def _get_dobj(r, X, beta, alpha, y=None):
        # the dual variable is theta = (y - X beta) / (alpha[0] * n_samples)
        n_samples = X.shape[0]
        theta = r / (alpha[0] * n_samples)
        dobj = alpha[0] * y @ theta
        dobj -= alpha[0] ** 2 * n_samples / 2 * np.dot(theta, theta)
        dobj -= alpha[0] ** 2 / alpha[1] / 2 * (ST(X.T @ theta, 1) ** 2).sum()
        return dobj

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask, :]

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
            mask_both = np.logical_and(mask0, mask)
            size_mat = mask.sum()
            dbeta0_new = np.zeros((size_mat, 2))
            count = 0
            count_old = 0
            n_features = mask.shape[0]
            for j in range(n_features):
                if mask_both[j]:
                    dbeta0_new[count, :] = jac0[count_old, :]
                if mask0[j]:
                    count_old += 1
                if mask[j]:
                    count += 1
        else:
            dbeta0_new = np.zeros((size_mat, 2))
        return dbeta0_new

    @staticmethod
    def _init_dbeta(n_features):
        dbeta = np.zeros((n_features, 2))
        return dbeta

    @staticmethod
    def _init_dr(dbeta, X, y, sign_beta, alpha):
        return - X @ dbeta

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return 0.0
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, r, dbeta, dr, L, alpha, beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            dbeta_old = dbeta[j, :].copy()
            dzj = dbeta[j, :] + Xs[:, j] @ dr / (L[j] * n_samples)
            dbeta[j:j+1, :] = (1 / (1 + alpha[1] / L[j])) * dzj

            dbeta[j:j+1, 0] -= (alpha[0] * np.sign(beta[j])
                                ) / L[j] / (1 + alpha[1] / L[j])
            dbeta[j:j+1, 1] -= (alpha[1] / L[j] * beta[j]
                                ) / (1 + alpha[1] / L[j])
            # update residuals
            dr[:, 0] -= Xs[:, j] * (dbeta[j, 0] - dbeta_old[0])
            dr[:, 1] -= Xs[:, j] * (dbeta[j, 1] - dbeta_old[1])

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, r, dr, L, alpha, beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            # store old beta j for fast update
            dbeta_old = dbeta[j, :].copy()
            dzj = dbeta[j, :] + Xjs @ dr[idx_nz, :] / (L[j] * n_samples)
            dbeta[j:j+1, :] = (1 / (1 + alpha[1] / L[j])) * dzj

            dbeta[j:j+1, 0] -= (alpha[0] * np.sign(beta[j])
                                ) / L[j] / (1 + alpha[1] / L[j])
            dbeta[j:j+1, 1] -= (alpha[1] / L[j] * beta[j]
                                ) / (1 + alpha[1] / L[j])
            # update residuals
            dr[idx_nz, 0] -= Xjs * (dbeta[j, 0] - dbeta_old[0])
            dr[idx_nz, 1] -= Xjs * (dbeta[j, 1] - dbeta_old[1])

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    @staticmethod
    def _get_jac_t_v(X, y, jac, mask, dense, alphas, v, n_samples):
        return np.array([alphas[0] * np.sign(dense) @ jac,
                         alphas[1] * dense @ jac])

    def proj_hyperparam(self, X, y, log_alpha):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(X.T @ y))
            alpha_max /= X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        if log_alpha[0] < self.log_alpha_max - 7:
            log_alpha[0] = self.log_alpha_max - 7
        elif log_alpha[0] > self.log_alpha_max + np.log(0.9):
            log_alpha[0] = self.log_alpha_max + np.log(0.9)
        if log_alpha[1] < self.log_alpha_max - 7:
            log_alpha[1] = self.log_alpha_max - 7
        elif log_alpha[1] > self.log_alpha_max + np.log(0.9):
            log_alpha[1] = self.log_alpha_max + np.log(0.9)
        return log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        # print(is_sparse)
        if is_sparse:
            return slinalg.norm(X, axis=0) ** 2 / (X.shape[0])
        else:
            return norm(X, axis=0) ** 2 / (X.shape[0])

    def _use_estimator(self, X, y, alpha, tol, max_iter):
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        self.estimator.set_params(
            tol=tol, alpha=alpha[0]+alpha[1],
            l1_ratio=alpha[0]/(alpha[0]+alpha[1]))
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
        return mask, dense, None

    @staticmethod
    def reduce_X(X, mask):
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        return y

    def sign(self, x, log_alpha):
        return x

    def get_beta(self, X, y, mask, dense):
        return mask, dense

    def get_jac_v(self, X, y, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    @staticmethod
    def get_hessian(X_train, y_train, mask, dense, log_alpha):
        n_samples = X_train.shape[0]
        hessian = np.exp(log_alpha[1]) * np.eye(mask.sum()) + \
            (1 / n_samples) * X_train[:, mask].T @ X_train[:, mask]
        return hessian

    def restrict_full_supp(self, X, y, mask, dense, v, log_alpha):
        return v

    def compute_alpha_max(self):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(self.X.T @ self.y))
            alpha_max /= self.X.shape[0]
            self.log_alpha_max = np.log(alpha_max)
        return self.log_alpha_max

    def get_jac_obj(self, Xs, ys, n_samples, beta, dbeta, r, dr, alpha):
        res1 = (1 / n_samples) * dr[:, 0].T @ dr[:, 0] + alpha[1] * \
            dbeta[:, 0].T @ dbeta[:, 0] + \
            alpha[0] * np.sign(beta) @ dbeta[:, 0]
        res2 = (1 / n_samples) * dr[:, 1].T @ dr[:, 1] + alpha[1] * \
            dbeta[:, 1].T @ dbeta[:, 1] + alpha[1] * beta @ dbeta[:, 1]
        return(norm(res2) + norm(res1))
