import numpy as np
from numpy.linalg import norm
from numba import njit
from scipy.sparse.linalg import LinearOperator
from sparse_ho.models import SVR
from sparse_ho.models.svr import (
    _compute_jac_aux, _compute_jac_aux_sparse,
    _update_beta_jac_bcd_aux, _update_beta_jac_bcd_aux_sparse)


class SimplexSVR(SVR):
    """The simplex support vector regression without bias
    The optimization problem is solved in the dual.

    It solves the SVR with probability vector constraints:
    sum_i beta_i = 1
    beta_i >= 0

    Parameters
    ----------
    estimator: sklearn
        An estimator that follows the scikit-learn API.
    """

    def __init__(self, estimator=None):
        super().__init__(estimator)

    def _init_dbeta_ddual_var(self, X, y, dense0=None,
                              mask0=None, jac0=None, compute_jac=True):
        n_samples, n_features = X.shape
        self.n_samples = n_samples
        ddual_var = np.zeros((2 * n_samples + n_features + 1, 2))

        if jac0 is None or not compute_jac or self.ddual_var is None:
            dbeta = np.zeros((n_features, 2))
        else:
            if self.ddual_var.shape[0] != 2 * n_samples + n_features + 1:
                dbeta = np.zeros((n_features, 2))
            else:
                ddual_var = self.ddual_var
                dbeta = X.T @ (
                    ddual_var[0:n_samples, :] -
                    ddual_var[n_samples:(2 * n_samples), :])
                dbeta += ddual_var[(2 * n_samples):
                                   (2 * n_samples + n_features), :]
                dbeta += ddual_var[-1, :]
        return dbeta, ddual_var

    def _init_beta_dual_var(self, X, y, mask0, dense0):
        n_samples, n_features = X.shape
        dual_var = np.zeros(2 * n_samples + n_features + 1)
        if mask0 is None or self.dual_var is None:
            beta = np.zeros(n_features)
        else:
            if self.dual_var.shape[0] != (2 * n_samples + n_features + 1):
                beta = np.zeros(n_features)
            else:
                dual_var = self.dual_var
                beta = X.T @ (dual_var[0:n_samples] -
                              dual_var[n_samples:(2 * n_samples)])
                beta += dual_var[(2 * n_samples):(2 * n_samples + n_features)]
                beta += dual_var[-1]
        return beta, dual_var

    @staticmethod
    @njit
    def _update_beta_jac_bcd(
            X, y, beta, dbeta, dual_var, ddual_var,
            hyperparam, L, compute_jac=True):

        C = hyperparam[0]
        epsilon = hyperparam[1]
        n_samples, n_features = X.shape
        for j in range(2 * n_samples + n_features + 1):
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples:
                    j1, j2, sign = j - n_samples, j, -1

                _update_beta_jac_bcd_aux(
                    X, y, epsilon, beta, dbeta, dual_var,
                    ddual_var, L, C / n_samples, j1, j2, sign, compute_jac)
            else:
                if j < (2 * n_samples + n_features):
                    F = beta[j - (2 * n_samples)]
                    dual_var_old = dual_var[j]
                    zj = dual_var[j] - F
                    dual_var[j] = max(0, zj)
                    beta[j - (2 * n_samples)] -= (dual_var_old - dual_var[j])
                    if compute_jac:
                        dF = dbeta[j - (2 * n_samples)]
                        ddual_var_old = ddual_var[j, :].copy()
                        dzj = ddual_var[j, :] - dF
                        if zj > 0:
                            ddual_var[j, :] = dzj
                        else:
                            ddual_var[j, :] = np.repeat(0.0, 2)
                        dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                          ddual_var[j, 0])
                        dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                          ddual_var[j, 1])
                else:
                    F = np.sum(beta) - 1
                    dual_var_old = dual_var[-1]
                    zj = dual_var[j] - F / n_features
                    dual_var[j] = zj
                    beta -= (dual_var_old - dual_var[-1])
                    if compute_jac:
                        dF = np.sum(dbeta, axis=0)
                        ddual_var_old = ddual_var[-1, :].copy()
                        dzj = ddual_var[j] - dF / n_features
                        ddual_var[j, :] = dzj
                        dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                        dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    @staticmethod
    @njit
    def _update_beta_jac_bcd_sparse(
            data, indptr, indices, y, n_samples, n_features, beta,
            dbeta, dual_var, ddual_var, hyperparam, L, compute_jac=True):
        C = hyperparam[0]
        epsilon = hyperparam[1]
        for j in range(2 * n_samples + n_features + 1):
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples and j < (2 * n_samples):
                    j1, j2, sign = j - n_samples, j, -1

                _update_beta_jac_bcd_aux_sparse(
                    data, indptr, indices, y, epsilon,
                    beta, dbeta, dual_var, ddual_var,
                    L, C / n_samples, j1, j2, sign, compute_jac)

            else:
                if j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                    F = beta[j - (2 * n_samples)]
                    dual_var_old = dual_var[j]
                    zj = dual_var[j] - F
                    dual_var[j] = max(0, zj)
                    beta[j - (2 * n_samples)] -= (dual_var_old - dual_var[j])
                    if compute_jac:
                        dF = dbeta[j - (2 * n_samples)]
                        ddual_var_old = ddual_var[j, :].copy()
                        dzj = ddual_var[j, :] - dF
                        if zj > 0:
                            ddual_var[j, :] = dzj
                        else:
                            ddual_var[j, :] = 0
                        dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                          ddual_var[j, 0])
                        dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                          ddual_var[j, 1])
                else:
                    F = np.sum(beta) - 1
                    dual_var_old = dual_var[-1]
                    zj = dual_var[j] - F / n_features
                    dual_var[j] = zj
                    beta -= (dual_var_old - dual_var[-1])
                    if compute_jac:
                        dF = np.sum(dbeta, axis=0)
                        ddual_var_old = ddual_var[-1, :].copy()
                        dzj = ddual_var[j, :] - dF / n_features
                        ddual_var[j, :] = dzj
                        dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                        dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    def _get_pobj0(self, dual_var, beta, hyperparam, y):
        hyperparameters = hyperparam.copy()
        hyperparameters[0] /= len(y)
        return super()._get_pobj0(dual_var, beta, hyperparameters, y)

    def _get_pobj(self, dual_var, X, beta, hyperparam, y):
        hyperparameters = hyperparam.copy()
        hyperparameters[0] /= len(y)
        return super()._get_pobj(dual_var, X, beta, hyperparameters, y)

    @staticmethod
    def _get_dobj(dual_var, X, beta, hyperparam, y):
        n_samples = X.shape[0]
        obj_dual = 0.5 * beta.T @ beta
        obj_dual += hyperparam[1] * np.sum(dual_var[0:(2 * n_samples)])
        obj_dual -= np.sum(y * (dual_var[0:n_samples] -
                                dual_var[n_samples:(2 * n_samples)]))
        obj_dual -= dual_var[-1]
        return -obj_dual

    @staticmethod
    def _get_jac(dbeta, mask):
        return dbeta[mask, :]

    def _init_dbeta0(self, mask, mask0, jac0):
        return super()._init_dbeta0(mask, mask0, jac0)

    def _init_dbeta(self, n_features):
        if self.dbeta is not None:
            return self.dbeta
        else:
            return np.zeros((n_features, 2))

    def _init_ddual_var(self, dbeta, X, y, sign_beta, hyperparam):
        dual_var = self.dual_var
        C = hyperparam[0]
        n_samples, n_features = X.shape
        sign = np.zeros(dual_var.shape[0])
        bool_temp = np.isclose(dual_var[0:(2 * n_samples + n_features)], 0.0)
        sign[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        sign[0:(2 * n_samples)][np.isclose(
            dual_var[0:(2 * n_samples)], C / n_samples)] = 1.0
        ddual_var = np.zeros((dual_var.shape[0], 2))
        if np.any(sign == 1.0):
            ddual_var[sign == 1.0, 0] = np.repeat(
                C / n_samples, (sign == 1).sum())
            ddual_var[sign == 1.0, 1] = np.repeat(
                0, (sign == 1).sum())
        self.ddual_var = ddual_var
        self.dbeta = X.T @ (
            ddual_var[0:n_samples, :] -
            ddual_var[n_samples:(2 * n_samples), :])
        self.dbeta += (
            ddual_var[(2 * n_samples):(2 * n_samples + n_features), :])
        self.dbeta += ddual_var[-1, :]
        return ddual_var

    @staticmethod
    @njit
    def _update_only_jac(X, y, dual_var, dbeta, ddual_var,
                         L, hyperparam, sign_beta):
        n_samples, n_features = X.shape
        length_dual = dual_var.shape[0]
        C = hyperparam[0]
        epsilon = hyperparam[1]
        gen_supp = np.zeros(length_dual)
        bool_temp = dual_var[0:(2 * n_samples + n_features)] == 0.0
        gen_supp[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        is_right_border = dual_var[0:(2 * n_samples)] == C / n_samples
        gen_supp[0:(2 * n_samples)][is_right_border] = 1.0
        for j in np.arange(0, length_dual)[gen_supp == 0.0]:
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples:
                    j1, j2, sign = j - n_samples, j, -1

                _compute_jac_aux(
                    X, epsilon, dbeta, ddual_var, dual_var[j2], L,
                    C / n_samples, j1, j2, sign)
            else:
                if j < (2 * n_samples + n_features):
                    dF = dbeta[j - (2 * n_samples)]
                    ddual_var_old = ddual_var[j, :].copy()
                    dzj = ddual_var[j, :] - dF
                    ddual_var[j, :] = dzj
                    dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                      ddual_var[j, 0])
                    dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                      ddual_var[j, 1])
                else:
                    dF = np.sum(dbeta, axis=0)
                    ddual_var_old = ddual_var[-1, :].copy()
                    dzj = ddual_var[j, :] - dF / n_features
                    ddual_var[j, :] = dzj
                    dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                    dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    @staticmethod
    @njit
    def _update_only_jac_sparse(
            data, indptr, indices, y, n_samples, n_features,
            dbeta, dual_var, ddual_var, L, hyperparam, sign_beta):
        C = hyperparam[0]
        epsilon = hyperparam[1]

        gen_supp = np.zeros(dual_var.shape[0])
        bool_temp = dual_var[0:(2 * n_samples + n_features)] == 0.0
        gen_supp[0:(2 * n_samples + n_features)][bool_temp] = -1.0
        is_right_border = dual_var[0:(2 * n_samples)] == C / n_samples
        gen_supp[0:(2 * n_samples)][is_right_border] = 1.0

        iter = np.arange(0, (2 * n_samples + n_features + 1))[gen_supp == 0.0]
        for j in iter:
            if j < (2 * n_samples):
                if j < n_samples:
                    j1, j2, sign = j, j, 1
                elif j >= n_samples:
                    j1, j2, sign = j - n_samples, j, -1

                _compute_jac_aux_sparse(
                    data, indptr, indices, epsilon, dbeta, ddual_var,
                    dual_var[j2], L, C / n_samples, j1, j2, sign)

            else:
                if j >= (2 * n_samples) and j < (2 * n_samples + n_features):
                    dF = dbeta[j - (2 * n_samples)]
                    ddual_var_old = ddual_var[j, :].copy()
                    dzj = ddual_var[j, :] - dF
                    ddual_var[j, :] = dzj
                    dbeta[j - (2 * n_samples), 0] -= (ddual_var_old[0] -
                                                      ddual_var[j, 0])
                    dbeta[j - (2 * n_samples), 1] -= (ddual_var_old[1] -
                                                      ddual_var[j, 1])
                else:
                    dF = np.sum(dbeta, axis=0)
                    ddual_var_old = ddual_var[-1, :].copy()
                    dzj = ddual_var[j, :] - dF / n_features
                    ddual_var[j, :] = dzj
                    dbeta[:, 0] -= (ddual_var_old[0] - ddual_var[-1, 0])
                    dbeta[:, 1] -= (ddual_var_old[1] - ddual_var[-1, 1])

    @staticmethod
    @njit
    def _reduce_alpha(alpha, mask):
        return alpha

    def get_L(self, X):
        """Compute Lipschitz constant of datafit.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.

        Returns
        -------
        L: float
            The Lipschitz constant.
        """
        return super().get_L(X)

    @staticmethod
    def reduce_X(X, mask):
        """Reduce design matrix to generalized support.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Design matrix.
        mask : ndarray, shape (n_features,)
            Generalized support.
        """
        return X[:, mask]

    @staticmethod
    def reduce_y(y, mask):
        """Reduce observation vector to generalized support.

        Parameters
        ----------
        y : ndarray, shape (n_samples,)
            Observation vector.
        mask : ndarray, shape (n_features,)  TODO shape n_samples right?
            Generalized support.
        """
        return y

    def sign(self, beta, log_hyperparams):
        """Get sign of iterate. Here sign means -1.0 if the iterate is 0,
        1.0 if it is equal to C / n_samples.

        Parameters
        ----------
        beta : ndarray, shape TODO
        log_hyperparams : ndarray, shape (2, )
            Logarithm of hyperparameter C and epsilon.
        """
        return super().sign(beta, log_hyperparams)

    def get_jac_v(self, X, y, mask, dense, jac, v):
        """Compute hypergradient.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: ndarray, shape (n_features,)
            Mask corresponding to non zero entries of beta.
        dense: ndarray, shape (mask.sum(),)
            Non zero entries of beta.
        jac: TODO
        v: TODO
        """
        return super().get_jac_v(X, y, mask, dense, jac, v)

    @staticmethod
    def get_full_jac_v(mask, jac_v, n_features):
        """TODO

        Parameters
        ----------
        mask: TODO
        jac_v: TODO
        n_features: int
            Number of features.
        """
        return jac_v

    def get_dual_v(self, mask, dense, X, y, v, log_hyperparam):
        """Compute the dual of v

        Parameters
        ----------
        mask: ndarray, shape (n_features,)
            Mask corresponding to non zero entries of beta.
        dense: ndarray, shape (mask.sum(),)
            Non zero entries of beta.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        v: TODO.

        log_hyperparam:
            ndarray, shape (2, )
            Logarithm of hyperparameter C and epsilon.
        """
        full_v = np.zeros(X.shape[1])
        full_v[mask] = v
        if v.shape[0] != 0:
            return np.hstack((-X @ full_v, full_v, np.sum(full_v)))
        else:
            return np.zeros(X.shape[0] + X.shape[1] + 1)

    def _get_grad(self, X, y, jac, mask, dense, hyperparam, v):
        C = hyperparam[0]
        n_samples = X.shape[0]
        epsilon = hyperparam[1]
        alpha = self.dual_var[0:n_samples] - \
            self.dual_var[n_samples:(2 * n_samples)]
        n_features = X.shape[1]
        gamma = self.dual_var[(2 * n_samples):(2 * n_samples + n_features)]
        mask0 = np.logical_not(np.isclose(gamma, 0))
        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))
        maskC = np.isclose(np.abs(alpha), C / n_samples)
        sub_id = np.zeros((mask0.sum(), n_features))
        sub_id[:, mask0] = 1.0

        hessian = np.concatenate((X[full_supp, :], -sub_id,
                                  -np.ones((1, n_features))), axis=0)
        hessian_vec = hessian @ X[maskC, :].T @ alpha[maskC]
        jac_t_v = -hessian_vec.T @ jac
        jac_t_v -= alpha[maskC].T @ v[0:n_samples][maskC]
        jac_t_v2 = -epsilon * np.sign(alpha[full_supp]) @ \
            jac[0:full_supp.sum()]
        return np.array([jac_t_v, jac_t_v2])

    def generalized_supp(self, X, v, log_hyperparam):
        """Generalized support of iterate.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        v: TODO.

        log_hyperparam: ndarray, shape (2, )
            Logarithm of hyperparameter C and epsilon.

        Returns
        -------
        TODO
        """
        n_samples, n_features = X.shape
        C = np.exp(log_hyperparam[0])
        alpha = self.dual_var[0:n_samples] -\
            self.dual_var[n_samples:(2 * n_samples)]
        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))
        mask0 = np.logical_not(np.isclose(
            self.dual_var[(2 * n_samples):(2 * n_samples + n_features)], 0))
        return v[np.hstack((full_supp, mask0, True))]

    def proj_hyperparam(self, X, y, log_hyperparam):
        """Project hyperparameter on an admissible range of values.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_hyperparam: ndarray, shape (2, )
            Logarithm of hyperparameters C and epsilon.

        Returns
        -------
        log_hyperparam: float
            Logarithm of projected hyperparameters.
        """
        return super().proj_hyperparam(X, y, log_hyperparam)

    def get_jac_residual_norm(self, Xs, ys, n_samples, sign_beta,
                              dbeta, dual_var, ddual_var, hyperparam):

        n_features = dbeta.shape[0]
        C = hyperparam[0]
        alpha = dual_var[0:n_samples] - dual_var[n_samples:(2 * n_samples)]
        dalpha = ddual_var[0:n_samples, 0] - \
            ddual_var[n_samples:(2 * n_samples), 0]
        dgamma = ddual_var[(2 * n_samples):(2 * n_samples + n_features), 0]
        dmu = ddual_var[-1, 0]
        maskC = np.isclose(np.abs(alpha), C / n_samples)

        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))

        vecX = dalpha[full_supp].T @ Xs[full_supp, :]
        vecX += dgamma + np.repeat(dmu, n_features)
        quadratic_term = vecX.T @ vecX
        linear_term = vecX.T @ Xs[maskC, :].T @ alpha[maskC]
        return norm(quadratic_term + linear_term)

    def get_mat_vec(self, X, y, mask, dense, log_C):
        """Returns a LinearOperator computing the matrix vector product
        with the Hessian of datafit. It is necessary to avoid storing a
        potentially large matrix, and keep advantage of the sparsity of X.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        mask: ndarray, shape (n_features,)
            Mask corresponding to non zero entries of beta.
        dense: ndarray, shape (mask.sum(),)
            Non zero entries of beta.
        log_C: ndarray
            Logarithm of hyperparameter.
        """
        C = np.exp(log_C)[0]
        n_samples, n_features = X.shape
        alpha = self.dual_var[0:n_samples] - \
            self.dual_var[n_samples:(2 * n_samples)]
        gamma = self.dual_var[(2 * n_samples):(2 * n_samples + n_features)]
        mask0 = np.logical_not(np.isclose(gamma, 0))
        full_supp = np.logical_not(
            np.logical_or(
                np.isclose(alpha, 0),
                np.isclose(np.abs(alpha), C / n_samples)))
        sub_id = np.zeros((mask0.sum(), n_features))
        sub_id[:, mask0] = 1.0
        X_m = np.concatenate((X[full_supp, :],
                              -sub_id, -np.ones((1, n_features))), axis=0)
        size_supp = X_m.shape[0]

        def mv(v):
            return X_m @ (X_m.T @ v)
        return LinearOperator((size_supp, size_supp), matvec=mv)

    def _use_estimator(self, X, y, hyperparam, tol, max_iter):
        return super()._use_estimator(X, y, hyperparam, tol, max_iter)
