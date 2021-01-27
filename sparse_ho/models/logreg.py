import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse, csc_matrix
from numba import njit

from sparse_ho.utils import init_dbeta0_new, ST, sigma, dual_logreg

from sparse_ho.models.base import BaseModel


class SparseLogreg(BaseModel):
    """Sparse Logistic Regression classifier.
    The objective function is:

    sum_1^n_samples log(1 + e^{-y_i x_i^T w}) + 1. / C * ||w||_1

    Parameters
    ----------
    X: {ndarray, sparse matrix} of (n_samples, n_features)
        Data.
    y: {ndarray, sparse matrix} of (n_samples)
        Target
    TODO: other parameters should be remove
    """

    def __init__(
            self, max_iter=1000, estimator=None, log_alpha_max=None):
        self.max_iter = max_iter
        self.log_alpha_max = log_alpha_max
        self.estimator = estimator

    def _init_dbeta_ddual_var(self, X, y, dense0=None,
                              mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        dbeta = np.zeros(n_features)
        if jac0 is None or not compute_jac:
            ddual_var = np.zeros(n_samples)
        else:
            dbeta[mask0] = jac0.copy()
            ddual_var = y * (X[:, mask0] @ jac0.copy())
        return dbeta, ddual_var

    def _init_beta_dual_var(self, X, y, mask0, dense0):
        beta = np.zeros(X.shape[1])
        if dense0 is None:
            dual_var = np.zeros(X.shape[0])
        else:
            beta[mask0] = dense0
            dual_var = y * (X[:, mask0] @ dense0)
        return beta, dual_var

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, dual_var, ddual_var,
            alpha, L, compute_jac=True):
        n_samples, n_features = X.shape
        for j in range(n_features):
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
                # compute derivatives
            sigmar = sigma(dual_var)
            grad_j = X[:, j] @ (y * (sigmar - 1))
            L_temp = np.sum(X[:, j] ** 2 * sigmar * (1 - sigmar))
            L_temp /= n_samples
            zj = beta[j] - grad_j / (L_temp * n_samples)
            beta[j] = ST(zj, alpha[j] / L_temp)
            dual_var += y * X[:, j] * (beta[j] - beta_old)
            if compute_jac:
                dsigmar = sigmar * (1 - sigmar) * ddual_var
                hess_fj = X[:, j] @ (y * dsigmar)
                dzj = dbeta[j] - hess_fj / (L_temp * n_samples)
                dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                dbeta[j:j+1] -= alpha[j] * np.sign(beta[j]) / L_temp
                # update residuals
                ddual_var += y * X[:, j] * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, dual_var, ddual_var, alphas, L, compute_jac=True):

        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero indices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            beta_old = beta[j]
            if compute_jac:
                dbeta_old = dbeta[j]
            sigmar = sigma(dual_var[idx_nz])
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
                    dsigmar = sigmar * (1 - sigmar) * ddual_var[idx_nz]
                    hess_fj = Xjs @ (y[idx_nz] * dsigmar)
                    dzj = dbeta[j] - hess_fj / (L_temp * n_samples)
                    dbeta[j:j+1] = np.abs(np.sign(beta[j])) * dzj
                    dbeta[j:j+1] -= alphas[j] * np.sign(beta[j]) / L_temp
                    # update residuals
                    ddual_var[idx_nz] += y[idx_nz] * Xjs * \
                        (dbeta[j] - dbeta_old)
                dual_var[idx_nz] += y[idx_nz] * Xjs * (beta[j] - beta_old)

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
            # TODO be careful r = y X beta
            r += X[:, j] * (beta[j-1] - beta[j])

        return grad

    @staticmethod
    def _get_pobj(dual_var, X, beta, alphas, y):
        pobj = (np.log1p(np.exp(- dual_var)).mean() +
                np.abs(alphas * beta).sum())
        return pobj

    @staticmethod
    def _get_dobj(dual_var, X, beta, alpha, y):
        n_samples = len(y)
        theta = y * sigma(- dual_var) / (alpha * n_samples)

        d_norm_theta = np.max(np.abs(X.T @ theta))
        if d_norm_theta > 1:
            theta /= d_norm_theta
        dobj = dual_logreg(y, theta, alpha)

        return dobj

    @staticmethod
    def _get_pobj0(r, beta, alphas, y):
        n_samples = r.shape[0]
        return np.log(2) / n_samples

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
    def _init_ddual_var(dbeta, X, y, sign_beta, alpha):
        return y * (X @ dbeta)

    def _init_g_backward(self, jac_v0):
        if jac_v0 is None:
            return 0.0
        else:
            return jac_v0

    @staticmethod
    @njit
    def _update_only_jac(Xs, y, dual_var, dbeta, ddual_var,
                         L, alpha, sign_beta):
        n_samples, n_features = Xs.shape
        for j in range(n_features):
            sigmar = sigma(dual_var)
            L_temp = np.sum(Xs[:, j] ** 2 * sigmar * (1 - sigmar))
            L_temp /= n_samples

            dbeta_old = dbeta[j]
            dsigmar = sigmar * (1 - sigmar) * ddual_var
            hess_fj = Xs[:, j] @ (y * dsigmar)
            dbeta[j:j+1] += - hess_fj / (L_temp * n_samples)
            dbeta[j:j+1] -= alpha * sign_beta[j] / L_temp
            # update residuals
            ddual_var += y * Xs[:, j] * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, dual_var, ddual_var, L, alpha, sign_beta):
        for j in range(n_features):
            # get the j-st column of X in sparse format
            Xjs = data[indptr[j]:indptr[j+1]]
            # get the non zero idices
            idx_nz = indices[indptr[j]:indptr[j+1]]
            sigmar = sigma(dual_var[idx_nz])
            L_temp = np.sum(Xjs ** 2 * sigmar * (1 - sigmar))
            L_temp /= n_samples
            if L_temp != 0:
                # store old beta j for fast update
                dbeta_old = dbeta[j]
                dsigmar = sigmar * (1 - sigmar) * ddual_var[idx_nz]

                hess_fj = Xjs @ (y[idx_nz] * dsigmar)
                # update of the Jacobian dbeta
                dbeta[j] -= hess_fj / (L_temp * n_samples)
                dbeta[j] -= alpha * sign_beta[j] / L_temp
                ddual_var[idx_nz] += y[idx_nz] * Xjs * (dbeta[j] - dbeta_old)

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    @staticmethod
    def _get_jac_t_v(X, y, jac, mask, dense, alphas, v, n_samples):
        return n_samples * alphas[mask] * np.sign(dense) @ jac

    def proj_hyperparam(self, X, y, log_alpha):
        if self.log_alpha_max is None:
            alpha_max = np.max(np.abs(X.T @ y))
            alpha_max /= (4 * X.shape[0])
            self.log_alpha_max = np.log(alpha_max)
        if log_alpha < self.log_alpha_max - 8:
            return self.log_alpha_max - 8
        elif log_alpha > self.log_alpha_max + np.log(0.9):
            return self.log_alpha_max + np.log(0.9)
        else:
            return log_alpha

    @staticmethod
    def get_L(X, is_sparse=False):
        return 0.0

    @staticmethod
    def reduce_X(X, mask):
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        return y

    def sign(self, x, log_alpha):
        return np.sign(x)

    def get_beta(self, X, y, mask, dense):
        return mask, dense

    def get_jac_v(self, X, y, mask, dense, jac, v):
        return jac.T @ v(mask, dense)

    @staticmethod
    def get_hessian(X, y, mask, dense, log_alpha):
        X_m = X[:, mask]
        a = y * (X_m @ dense)
        temp = sigma(a) * (1 - sigma(a))
        is_sparse = issparse(X)
        if is_sparse:
            hessian = csc_matrix(
                X_m.T.multiply(temp)) @ X_m
        else:
            hessian = (X_m.T * temp) @ X_m
        return hessian

    def generalized_supp(self, X, v, log_alpha):
        return v

    def compute_alpha_max(self, X, y):
        alpha_max = np.max(np.abs(X.T @ y))
        alpha_max /= (4 * X.shape[0])
        log_alpha_max = np.log(alpha_max)
        return log_alpha_max

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta, dbeta, dual_var,
                    ddual_var, alpha):
        return(
            norm(ddual_var.T @ ddual_var +
                 n_samples * alpha * sign_beta @ dbeta))

    def _use_estimator(self, X, y, alpha, tol, max_iter):
        n_samples = X.shape[0]
        if self.estimator is None:
            raise ValueError("You did not pass a solver with sklearn API")
        self.estimator.set_params(tol=tol, C=1/(alpha*n_samples))
        self.estimator.max_iter = self.max_iter
        self.estimator.fit(X, y)
        mask = self.estimator.coef_ != 0
        dense = self.estimator.coef_[mask]
        return mask[0], dense, None
