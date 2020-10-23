from numpy.linalg import norm
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import check_cv
from sklearn.utils import check_random_state

from sparse_ho.utils import sigma, smooth_hinge
from sparse_ho.utils import derivative_smooth_hinge
from sparse_ho.forward import get_beta_jac_iterdiff


class CV():
    """Held out loss for quadratic datafit (we should change the name CV here).

    Attributes
    ----------
    TODO
    """
    # XXX : this code should be the same as CrossVal as you can pass
    # cv as [(train, test)] ie directly the indices of the train
    # and test splits.

    def __init__(self, X_val, y_val, model, convexify=False,
                 gamma_convex=1e-2, X_test=None, y_test=None):
        """
        Parameters
        ----------
        X_val : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Validation data
        y_val : {ndarray, sparse matrix} of shape (n_samples,)
            Validation target
        model: object of the class Model (e.g. Lasso or SparseLogreg)
        X_test : {ndarray, sparse matrix} of shape (n_samples_test, n_features)
            Test data
        convexify: bool
            this param should be remove from here XXX
        gamma_convex: bool
            this param should be removed from here XXX
        y_test : {ndarray, sparse matrix} of (n_samples_test)
            Test target
        """
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.convexify = convexify
        self.gamma_convex = gamma_convex

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.val_test = None
        self.rmse = None

    def get_v(self, mask, dense):
        return 2 * (self.X_val[:, mask].T @ (
            self.X_val[:, mask] @ dense - self.y_val)) / self.X_val.shape[0]

    def value(self, mask, dense):
        val = (
            norm(self.y_val - self.X_val[:, mask] @ dense) ** 2 /
            self.X_val.shape[0])
        return val

    def value_test(self, mask, dense):
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

    def get_val(self, log_alpha, tol=1e-3):
        # TODO add warm start
        mask, dense, _ = get_beta_jac_iterdiff(
            self.model.X, self.model.y, log_alpha, self.model, tol=tol, compute_jac=False)
        self.value_test(mask, dense)
        return self.value(mask, dense)

    def get_val_grad(
            self, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, backward=False, beta_star=None):
        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            self.model.X, self.model.y, log_alpha, self.model, self.get_v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            backward=backward, full_jac_v=True)
        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        mask, dense = self.model.get_primal(mask, dense)
        val = self.value(mask, dense)
        self.value_test(mask, dense)
        self.compute_rmse(mask, dense, beta_star)
        if self.convexify:
            val += self.gamma_convex + np.sum(np.exp(log_alpha) ** 2)
            grad += 2 * self.gamma_convex * np.exp(log_alpha)
        return val, grad


class Logistic():
    """Logistic loss.
    """
    def __init__(self, X_val, y_val, model, X_test=None, y_test=None):
        """
        Parameters
        ----------
        X_val : {ndarray, sparse matrix} of (n_samples, n_features)
            Validation data
        y_val : {ndarray, sparse matrix} of (n_samples)
            Validation target
        model: object of the class Model (e.g. Lasso or Sparse logistic regression)
        X_test : {ndarray, sparse matrix} of (n_samples_test, n_features)
            Test data
        y_test : {ndarray, sparse matrix} of (n_samples_test)
            Test target
        """
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.val_test = None
        self.rmse = None

    def get_v(self, mask, dense):
        temp = sigma(self.y_val * (self.X_val[:, mask] @ dense))
        v = self.X_val[:, mask].T @ (self.y_val * (temp - 1))
        v /= self.X_val.shape[0]
        return v

    def value(self, mask, dense):
        val = np.sum(
            np.log(1 + np.exp(-self.y_val * (self.X_val[:, mask] @ dense))))
        val /= self.X_val.shape[0]
        return val

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

    def get_val(self, log_alpha, tol=1e-3):
        # TODO add warm start
        mask, dense, _ = get_beta_jac_iterdiff(
            self.model.X, self.model.y, log_alpha, self.model, tol=tol, compute_jac=False)
        self.value_test(mask, dense)
        return self.value(mask, dense)

    def get_val_grad(
            self, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, backward=False, beta_star=None):
        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            self.model.X, self.model.y, log_alpha, self.model, self.get_v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            backward=backward, full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        mask, dense = self.model.get_primal(mask, dense)
        val = self.value(mask, dense)
        self.value_test(mask, dense)
        self.compute_rmse(mask, dense, beta_star)

        return val, grad


class SmoothedHinge():
    """Smooth Hinge loss.

    Attributes
    ----------
    TODO
    """
    def __init__(self, X_val, y_val, model, X_test=None, y_test=None):
        """
        Parameters
        ----------
        X_val : {ndarray, sparse matrix} of shape (n_samples, n_features)
            Validation data
        y_val : {ndarray, sparse matrix} of shape (n_samples)
            Validation target
        model: instance of Model
            Object of the class Model (e.g. Lasso or Sparse logistic regression)
        X_test : {ndarray, sparse matrix} of shape (n_samples_test, n_features)
            Test data
        y_test : {ndarray, sparse matrix} of shape (n_samples_test,)
            Test target
        """
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.val_test = None
        self.rmse = None

    def get_v(self, mask, dense):
        Xbeta_y = self.y_val * (self.X_val[:, mask] @ dense)
        deriv = derivative_smooth_hinge(Xbeta_y)
        if issparse(self.X_val):
            v = self.X_val[:, mask].T.multiply(deriv * self.y_val)
            v = np.array(np.sum(v, axis=1))
            v = np.squeeze(v)
        else:
            v = (deriv * self.y_val)[:, np.newaxis] * self.X_val[:, mask]
            v = np.sum(v, axis=0)
        v /= self.X_val.shape[0]
        return v

    def value(self, mask, dense):
        if issparse(self.X_val):
            Xbeta_y = (self.X_val[:, mask].T).multiply(self.y_val).T @ dense
        else:
            Xbeta_y = self.y_val * (self.X_val[:, mask] @ dense)
        return np.sum(smooth_hinge(Xbeta_y)) / self.X_val.shape[0]

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
            self, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, backward=False, beta_star=None):
        mask, dense, grad, quantity_to_warm_start = get_beta_jac_v(
            self.model.X, self.model.y, log_alpha, self.model, self.get_v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            backward=backward, full_jac_v=True)

        self.mask0 = mask
        self.dense0 = dense
        self.quantity_to_warm_start = quantity_to_warm_start
        mask, dense = self.model.get_primal(mask, dense)
        val = self.value(mask, dense)
        self.value_test(mask, dense)
        self.compute_rmse(mask, dense, beta_star)

        return val, grad

    def get_val(self, log_alpha, tol=1e-3):
        mask, dense, _ = get_beta_jac_iterdiff(
            self.model.X, self.model.y, log_alpha, self.model,
            max_iter=self.model.max_iter, tol=tol, compute_jac=False)
        mask, dense = self.model.get_primal(mask, dense)
        val = self.value(mask, dense)
        self.value_test(mask, dense)
        return val


class SURE():
    """Stein Unbiased Risk Estimator (SURE).

    Attributes
    ----------
    TODO
    """
    def __init__(self, X, y, model, sigma, C=2.0,
                 gamma_sure=0.3, random_state=42,
                 X_test=None, y_test=None):
        """
        Parameters
        ----------
        X_ : {ndarray, sparse matrix} of (n_samples, n_features)
            Validation data
        y : {ndarray, sparse matrix} of (n_samples)
            Validation target
        model: instance of Model
            The model (e.g. instance of Lasso or SparseLogreg)
        sigma: float
            Noise level
        random_state : int, RandomState instance, default=42
            The seed of the pseudo random number generator.
            Pass an int for reproducible output across multiple function calls.
        X_test, y_test: TODO we should remove these parameters no? -> YES !
        """
        self.X_val = X
        self.y_val = y
        self.model = model
        self.sigma = sigma
        self.C = C
        self.gamma_sure = gamma_sure
        self.epsilon = C * sigma / (X.shape[0]) ** gamma_sure
        rng = check_random_state(random_state)
        self.delta = rng.randn(X.shape[0])  # sample random noise for MCMC step

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None

        self.mask02 = None
        self.dense02 = None
        self.quantity_to_warm_start2 = None

        self.val_test = None
        self.rmse = None

    def v(self, mask, dense):
        return (2 * self.X_val[:, mask].T @ (
                self.X_val[:, mask] @ dense - self.y_val -
                self.delta * self.sigma ** 2 / self.epsilon))

    def v2(self, mask, dense):
        return ((2 * self.sigma ** 2 *
                self.X_val[:, mask].T @ self.delta / self.epsilon))

    def value(self, mask, dense, mask2, dense2):
        dof = ((self.X_val[:, mask2] @ dense2 -
                self.X_val[:, mask] @ dense) @ self.delta)
        dof /= self.epsilon
        # compute the value of the sure
        val = norm(self.y_val - self.X_val[:, mask] @ dense) ** 2
        val -= self.X_val.shape[0] * self.sigma ** 2
        val += 2 * self.sigma ** 2 * dof

        return val

    def value_test(self, mask, dense):
        # self.val_test = None
        val = (
            norm(self.y_val - self.X_val[:, mask] @ dense) ** 2 /
            self.X_val.shape[0])
        self.val_test = val
        return val

    def compute_rmse(self, mask, dense, beta_star):
        if beta_star is not None:
            diff_beta = beta_star.copy()
            diff_beta[mask] -= dense
            self.rmse = norm(diff_beta)
        else:
            self.rmse = None

    def get_val(self, log_alpha, tol=1e-3):
        # TODO add warm start
        mask, dense, _ = get_beta_jac_iterdiff(
            self.model.X, self.model.y, log_alpha, self.model,
            tol=tol, mask0=self.mask0, dense0=self.dense0, compute_jac=False)
        mask2, dense2, _ = get_beta_jac_iterdiff(
            self.model.X, self.model.y + self.epsilon * self.delta,
            log_alpha, self.model,
            tol=tol, compute_jac=False)

        val = self.value(mask, dense, mask2, dense2)

        return val

    def get_val_grad(
            self, log_alpha, get_beta_jac_v,
            mask0=None, dense0=None, beta_star=None,
            jac0=None, max_iter=1000, tol=1e-3, compute_jac=True,
            backward=False):
        mask, dense, jac_v, quantity_to_warm_start = get_beta_jac_v(
            self.model.X, self.model.y, log_alpha, self.model, self.v,
            mask0=self.mask0, dense0=self.dense0,
            quantity_to_warm_start=self.quantity_to_warm_start,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            backward=backward, full_jac_v=True)
        mask2, dense2, jac_v2, quantity_to_warm_start2 = get_beta_jac_v(
            self.model.X, self.model.y + self.epsilon * self.delta,
            log_alpha, self.model, self.v2, mask0=self.mask02,
            dense0=self.dense02,
            quantity_to_warm_start=self.quantity_to_warm_start2,
            max_iter=max_iter, tol=tol, compute_jac=compute_jac,
            backward=backward, full_jac_v=True)
        val = self.value(mask, dense, mask2, dense2)
        self.value_test(mask, dense)
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

            model = Model(
                X_train, y_train, max_iter=max_iter, estimator=estimator)

            criterion = CV(
                X_val, y_val, model, X_test=X_val, y_test=y_val)

            self.dict_crits[i] = criterion
        self.n_splits = cv.n_splits
        self.model = self.dict_crits[0].model

    def get_val(self, log_alpha, tol=1e-3):
        val = 0
        for i in range(self.n_splits):
            vali = self.dict_crits[i].get_val(log_alpha, tol=tol)
            val += vali
        val /= self.n_splits
        return val

    def get_val_grad(
            self, log_alpha, get_beta_jac_v, max_iter=10000, tol=1e-5,
            compute_jac=True, backward=False, beta_star=None):
        val = 0
        grad = 0
        for i in range(self.n_splits):
            vali, gradi = self.dict_crits[i].get_val_grad(
                log_alpha, get_beta_jac_v, max_iter=max_iter, tol=tol,
                compute_jac=compute_jac, backward=backward,
                beta_star=beta_star)
            val += vali
            if gradi is not None:
                grad += gradi
        val /= self.n_splits
        if gradi is not None:
            grad /= self.n_splits
        else:
            grad = None
        return val, grad


class LogisticMulticlass():
    """Multiclass logistic loss.
    """
    def __init__(self, X_val, y_val, model, X_test=None, y_test=None):
        """
        Parameters
        ----------
        X_val : {ndarray, sparse matrix} of (n_samples, n_features)
            Validation data
        y_val : {ndarray, sparse matrix} of (n_samples)
            Validation target
        model: object of the class Model (e.g. Lasso or Sparse logistic regression)
        X_test : {ndarray, sparse matrix} of (n_samples_test, n_features)
            Test data
        y_test : {ndarray, sparse matrix} of (n_samples_test)
            Test target
        """
        self.X_val = X_val
        self.y_val = y_val
        self.X_test = X_test
        self.y_test = y_test
        self.model = model

        self.mask0 = None
        self.dense0 = None
        self.quantity_to_warm_start = None
        self.val_test = None
        self.rmse = None

    def value(self, mask, dense):
        """TODO
        """
        return None
