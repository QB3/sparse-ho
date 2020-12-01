import numpy as np
from numpy.linalg import norm
from numba import njit
from sparse_ho.utils import ST, init_dbeta0_new
from sparse_ho.utils import sigma
from scipy.sparse import issparse, csc_matrix


class SparseLogreg():
    r"""Sparse Logistic Regression classifier.
    The objective function is:

    ..math::

        1/n_samples \sum_i log(1 + e^{-y_i x_i^T w}) + 1. / C * ||w||_1

    Parameters
    ----------
    max_iter: int
        Defaults 1000.
        number of maximum iteration for the Lasso resolution
    estimator: instance of ``sklearn.base.BaseEstimator``
        An estimator that follows the scikit-learn API.
    log_alpha_max: float
        logarithm of alpha_max if already precomputed
    """

    def __init__(
            self, max_iter=1000, estimator=None, log_alpha_max=None):
        self.max_iter = max_iter
        self.log_alpha_max = log_alpha_max
        self.estimator = estimator

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
    def _get_pobj(r, X, beta, alphas, y):
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
    def _init_dr(dbeta, X, y, sign_beta, alpha):
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

    def restrict_full_supp(self, X, y, mask, dense, v, log_alpha):
        return v

    def compute_alpha_max(self, X, y):
        alpha_max = np.max(np.abs(X.T @ y))
        alpha_max /= (4 * X.shape[0])
        log_alpha_max = np.log(alpha_max)
        return log_alpha_max

    def get_jac_obj(self, Xs, ys, n_samples, sign_beta, dbeta, r, dr, alpha):
        return(
            norm(dr.T @ dr + n_samples * alpha * sign_beta @ dbeta))

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
