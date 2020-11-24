from numpy.linalg import norm
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state

from sparse_ho.utils import sigma, smooth_hinge
from sparse_ho.utils import derivative_smooth_hinge
from sparse_ho.forward import get_beta_jac_iterdiff


class HeldOutMSE():
    """Held out loss for quadratic datafit.

    Attributes
    ----------
    TODO
    """
    # XXX : this code should be the same as CrossVal as you can pass
    # cv as [(train, test)] ie directly the indices of the train
    # and test splits.

    def __init__(
            self, idx_train, idx_val, X_test=None, y_test=None,):
        """
        Parameters
        ----------
            Validation target
        X_test : {ndarray, sparse matrix} of shape (n_samples_test, n_features)
            Test data
        y_test : {ndarray, sparse matrix} of (n_samples_test)
            Test target
        """
        self.X_test = X_test
        self.y_test = y_test
        self.idx_train = idx_train
        self.idx_val = idx_val

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.val_test = None
        self.rmse = None

    def get_mse_val(self, X, y, mask, dense):
        """Compute the MSE on the validation set.
        """
        val = (
            norm(y[self.idx_val] - X[np.ix_(self.idx_val, mask)] @ dense) ** 2 /
            len(self.idx_val))
        return val

    def get_mse_test(self, mask, dense):
        """Compute the MSE on the validation set.
        TODO
        """
        if self.X_test is not None and self.y_test is not None:
            self.val_test = (
                norm(self.y_test - self.X_test[:, mask] @ dense) ** 2 /
                self.X_test.shape[0])
        else:
            self.val_test = None

    def compute_rmse(self, mask, dense, beta_star):
        if beta_star is not None:
            diff_beta = beta_star.copy()
            diff_beta[mask] -= dense
            self.rmse = norm(diff_beta)
        else:
            self.rmse = None

    def get_val(self, model, X, y, log_alpha, tol=1e-3):
        # TODO add warm start
        mask, dense, _ = get_beta_jac_iterdiff(
            X[self.idx_train], y[self.idx_train], log_alpha, model, tol=tol, compute_jac=False)
        self.get_mse_test(mask, dense)
        return self.get_mse_val(mask, dense)

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, beta_star=None):

        def get_v(mask, dense):
            return 2 * (
                X[np.ix_(self.idx_val, mask)].T @ (
                    X[np.ix_(self.idx_val, mask)] @ dense - y[self.idx_val])) / len(self.idx_val)

        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            X[self.idx_train, :], y[self.idx_train], log_alpha, model,
            get_v, mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac, full_jac_v=True)
        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        mask, dense = model.get_primal(mask, dense)
        val = self.get_mse_val(X, y, mask, dense)
        # TODO put the following in a callback function
        self.get_mse_test(mask, dense)
        self.compute_rmse(mask, dense, beta_star)

        return val, grad


class HeldOutLogistic():
    """Logistic loss on held out data
    """

    def __init__(self, idx_train, idx_val, X_test=None, y_test=None):
        """
        Parameters
        ----------
        X_val : {ndarray, sparse matrix} of (n_samples, n_features)
            Validation data
        y_val : {ndarray, sparse matrix} of (n_samples)
            Validation target
        X_test : {ndarray, sparse matrix} of (n_samples_test, n_features)
            Test data
        y_test : {ndarray, sparse matrix} of (n_samples_test)
            Test target
        """
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.X_test = X_test
        self.y_test = y_test

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.val_test = None
        self.rmse = None

    @staticmethod
    def value_outer_crit(X, y, mask, dense):
        if X is not None and y is not None:
            val = np.sum(
                np.log(1 + np.exp(-y * (X[:, mask] @ dense))))
            val /= X.shape[0]
            return val
        else:
            return None

    def value_test(self, mask, dense):
        if self.X_test is not None and self.y_test is not None:
            self.val_test = np.sum(
                np.log(1 + np.exp(-self.y_test * (self.X_test[:, mask] @ dense))))
            self.val_test /= self.X_test.shape[0]
        else:
            self.val_test = None

    def compute_rmse(self, mask, dense, beta_star):
        if beta_star is not None:
            diff_beta = beta_star.copy()
            diff_beta[mask] -= dense
            self.rmse = norm(diff_beta)
        else:
            self.rmse = None

    def get_val(self, model, log_alpha, tol=1e-3):
        # TODO add warm start
        # TODO on train or on test ?
        mask, dense, _ = get_beta_jac_iterdiff(
            X[self.idx_val], y[self.idx_val], log_alpha, model, tol=tol, compute_jac=False)
        self.value_test(mask, dense)
        return self.value(mask, dense)

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, beta_star=None):

        def get_v(mask, dense):
            temp = sigma(y * (X[:, mask] @ dense))
            v = X[:, mask].T @ (y * (temp - 1))
            v /= X.shape[0]
            return v

        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            X[self.idx_train, :], y[self.idx_train], log_alpha, model,
            get_v, mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        mask, dense = model.get_primal(
            X[self.idx_train, :], y[self.idx_train], mask, dense)
        val = self.value_outer_crit(
            X[self.idx_val, :], y[self.idx_val], mask, dense)
        self.compute_rmse(mask, dense, beta_star)

        return val, grad


class HeldOutSmoothedHinge():
    """Smooth Hinge loss.

    Attributes
    ----------
    TODO
    """

    def __init__(self, idx_train, idx_val, X_test=None, y_test=None):
        """
        Parameters
        X_test : {ndarray, sparse matrix} of shape (n_samples_test, n_features)
            Test data
        y_test : {ndarray, sparse matrix} of shape (n_samples_test,)
            Test target
        """
        self.idx_train = idx_train
        self.idx_val = idx_val
        self.X_test = X_test
        self.y_test = y_test

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.val_test = None
        self.rmse = None

    def value(self, X_val, y_val, mask, dense):
        if issparse(X_val):
            Xbeta_y = (X_val[:, mask].T).multiply(y_val).T @ dense
        else:
            Xbeta_y = y_val * (X_val[:, mask] @ dense)
        return np.sum(smooth_hinge(Xbeta_y)) / X_val.shape[0]

    def value_test(self, mask, dense):
        if self.X_test is not None and self.y_test is not None:
            if issparse(self.X_test):
                Xbeta_y = (self.X_test[:, mask].T).multiply(self.y_test).T @ dense
            else:
                Xbeta_y = self.y_test * (self.X_test[:, mask] @ dense)
            self.val_test = np.sum(smooth_hinge(Xbeta_y)) / self.X_test.shape[0]
        else:
            self.val_test = None

    def compute_rmse(self, mask, dense, beta_star):
        if beta_star is not None:
            diff_beta = beta_star.copy()
            diff_beta[mask] -= dense
            self.rmse = norm(diff_beta)
        else:
            self.rmse = None

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, beta_star=None):
        def get_v(mask, dense):
            Xbeta_y = y[self.idx_val] * (
                X[np.ix_(self.idx_val, mask)] @ dense)
            deriv = derivative_smooth_hinge(Xbeta_y)
            if issparse(X):
                v = X[np.ix_(self.idx_val, mask)].T.multiply(deriv * y[self.idx_val])
                v = np.array(np.sum(v, axis=1))
                v = np.squeeze(v)
            else:
                v = (deriv * y[self.idx_val])[:, np.newaxis] * X[np.ix_(self.idx_val, mask)]
                v = np.sum(v, axis=0)
            v /= len(self.idx_val)
            return v
        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            X[self.idx_train], y[self.idx_train], log_alpha, model, get_v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        mask, dense = model.get_primal(
            X[self.idx_train, :], y[self.idx_train], mask, dense)
        val = self.value(X[self.idx_val], y[self.idx_val], mask, dense)
        # self.value_test(mask, dense)
        # self.compute_rmse(mask, dense, beta_star)

        return val, grad

    def get_val(self, model, log_alpha, tol=1e-3):
        # TODO X y also ?
        mask, dense, _ = get_beta_jac_iterdiff(
            model.X, model.y, log_alpha, model,  # TODO max_iter
            max_iter=model.max_iter, tol=tol, compute_jac=False)
        mask, dense = model.get_primal(mask, dense)
        val = self.value(mask, dense)
        self.value_test(mask, dense)
        return val


class SmoothedSURE():
    """Smoothed version of the Stein Unbiased Risk Estimator (SURE).

    Implements the iterative Finite-Difference Monte-Carlo approximation of the
    SURE. By default, the approximation is ruled by a power law heuristic [1].

    Attributes
    ----------
    TODO

    References
    ----------
    .. [1] C.-A. Deledalle, Stein Unbiased GrAdient estimator of the Risk (SUGAR)
    for multiple parameter selection. SIAM J. Imaging Sci., 7(4), 2448-2487.
    """

    def __init__(self, sigma, finite_difference_step=None,
                 random_state=42):
        """
        Parameters
        ----------
        sigma: float
            Noise level
        finite_difference_step: float, optional
            Finite difference step used in the approximation of the SURE.
            By default, use a power law heuristic.
        random_state : int, RandomState instance, default=42
            The seed of the pseudo random number generator.
            Pass an int for reproducible output across multiple function calls.
        """
        self.sigma = sigma
        self.random_state = random_state
        self.finite_difference_step = finite_difference_step
        self.init_delta_epsilon = False

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None

        self.mask02 = None
        self.dense02 = None
        self.quantity_to_warm_start2 = None

        self.val_test = None
        self.rmse = None

    def value(self, X, y, mask, dense, mask2, dense2):
        dof = ((X[:, mask2] @ dense2 -
                X[:, mask] @ dense) @ self.delta)
        dof /= self.epsilon
        # compute the value of the sure
        val = norm(y - X[:, mask] @ dense) ** 2
        val -= X.shape[0] * self.sigma ** 2
        val += 2 * self.sigma ** 2 * dof

        return val

    def value_test(self, X, y, mask, dense):
        val = (
            norm(y - X[:, mask] @ dense) ** 2 / X.shape[0])
        self.val_test = val
        return val

    def compute_rmse(self, mask, dense, beta_star):
        if beta_star is not None:
            diff_beta = beta_star.copy()
            diff_beta[mask] -= dense
            self.rmse = norm(diff_beta)
        else:
            self.rmse = None

    def get_val(self, model, X, y, log_alpha, tol=1e-3):
        # TODO add warm start
        mask, dense, _ = get_beta_jac_iterdiff(
            X[self.idx_train], y[self.idx_train], log_alpha, model,
            tol=tol, mask0=self.mask0, dense0=self.dense0, compute_jac=False)
        mask2, dense2, _ = get_beta_jac_iterdiff(
            X[self.idx_train], y[self.idx_train] + self.epsilon * self.delta,
            log_alpha, model, tol=tol, compute_jac=False)

        val = self.value(mask, dense, mask2, dense2)

        return val

    def _init_delta_epsilon(self, X):
        if self.finite_difference_step:
            self.epsilon = self.finite_difference_step
        else:
            # Use Deledalle et al. 2014 heuristic
            self.epsilon = 2.0 * self.sigma / (X.shape[0]) ** 0.3
        rng = check_random_state(self.random_state)
        self.delta = rng.randn(X.shape[0])  # sample random noise for MCMC step
        self.init_delta_epsilon = True

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v,
            mask0=None, dense0=None, beta_star=None,
            jac0=None, max_iter=1000, tol=1e-3, compute_jac=True):
        if not self.init_delta_epsilon:
            self._init_delta_epsilon(X)

        def v(mask, dense):
            return (2 * X[:, mask].T @ (
                    X[:, mask] @ dense - y -
                    self.delta * self.sigma ** 2 / self.epsilon))

        def v2(mask, dense):
            return ((2 * self.sigma ** 2 *
                     X[:, mask].T @ self.delta / self.epsilon))

        mask, dense, jac_v, quantity_to_warm_start = get_beta_jac_v(
            X, y, log_alpha, model, v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            full_jac_v=True)
        mask2, dense2, jac_v2, quantity_to_warm_start2 = get_beta_jac_v(
            X, y + self.epsilon * self.delta,
            log_alpha, model, v2, mask0=self.mask02,
            dense0=self.dense02,
            quantity_to_warm_start=self.quantity_to_warm_start2,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            full_jac_v=True)
        val = self.value(X, y, mask, dense, mask2, dense2)
        self.value_test(X, y, mask, dense)
        self.compute_rmse(mask, dense, beta_star)
        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start

        self.mask02 = mask2
        self.dense02 = dense2
        self.quantity_to_warm_start2 = quantity_to_warm_start2

        if jac_v is not None and jac_v2 is not None:
            grad = jac_v + jac_v2
        else:
            grad = None

        return val, grad


class CrossVal():
    """Cross-validation loss.

    Attributes
    ----------
    dict_crits : dict
        The instances of criterion used for each fold.
    val_test : None
        XXX
    rmse : None
        XXX
    """

    def __init__(self, X, y, Model, cv=None, max_iter=1000, estimator=None):
        """
        Parameters
        ----------
        X : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Data
        y : {ndarray, sparse matrix} of shape (n_samples,)
            Target
        Model: class
            The Model class definition (e.g. Lasso or SparseLogreg)
        cv : int, cross-validation generator or iterable, default=None
            Determines the cross-validation splitting strategy.
            Possible inputs for cv are:

            - None, to use the default 5-fold cross-validation,
            - int, to specify the number of folds.
            - scikit-learn CV splitter
            - An iterable yielding (train, test) splits as arrays of indices.

            For int/None inputs, KFold is used.
        max_iter: int
            Maximal number of iteration for the state-of-the-art solver
        estimator: instance of ``sklearn.base.BaseEstimator``
            An estimator that follows the scikit-learn API.
        """
        self.X = X
        self.y = y
        self.dict_crits = {}
        self.dict_models = {}
        self.val_test = None
        self.rmse = None
        self.estimator = estimator

        cv = check_cv(cv)

        for i, (train, val) in enumerate(cv.split(X)):
            X_train = X[train, :]
            y_train = y[train]
            X_val = X[val, :]
            y_val = y[val]

            if issparse(X_train):
                X_train = X_train.tocsc()
            if issparse(X_val):
                X_val = X_val.tocsc()

            # TODO get rid of this
            self.models[i] = Model(
                X_train, y_train, max_iter=max_iter, estimator=estimator)

            criterion = HeldOutMSE(
                X_val, y_val, X_test=X_val, y_test=y_val)

            self.dict_crits[i] = criterion
        self.n_splits = cv.n_splits

    def get_val(self, model, log_alpha, tol=1e-3):
        val = 0
        for i in range(self.n_splits):
            vali = self.dict_crits[i].get_val(self.models[i], log_alpha,
                                              tol=tol)
            val += vali
        val /= self.n_splits
        return val

    def get_val_grad(
            self, model, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, beta_star=None):
        val = 0
        grad = 0
        for i in range(self.n_splits):
            vali, gradi = self.dict_crits[i].get_val_grad(
                self.models[i], log_alpha, get_beta_jac_v, max_iter=max_iter,
                tol=tol,
                compute_jac=compute_jac, beta_star=beta_star)
            val += vali
            if gradi is not None:
                grad += gradi
        val /= self.n_splits
        if gradi is not None:
            grad /= self.n_splits
        else:
            grad = None
        return val, grad
