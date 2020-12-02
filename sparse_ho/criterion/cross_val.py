from numpy.linalg import norm
import numpy as np
from scipy.sparse import issparse
from sklearn.model_selection import check_cv

from sparse_ho.algo.forward import get_beta_jac_iterdiff
from sparse_ho.criterion.base import BaseCriterion


class CrossVal(BaseCriterion):
    """Cross-validation loss.

    Attributes
    ----------
    dict_crits : dict
        The instances of criterion used for each fold.
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
        self.rmse = None
        self.estimator = estimator

        cv = check_cv(cv)

        for i, (idx_train, idx_val) in enumerate(cv.split(X)):
            X_train = X[idx_train, :]
            y_train = y[idx_train]

            if issparse(X_train):
                X_train = X_train.tocsc()

            # TODO get rid of this
            self.models[i] = Model(
                X_train, y_train, max_iter=max_iter, estimator=estimator)

            criterion = HeldOutMSE(idx_train, idx_val)

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
            compute_jac=True, monitor=None):
        val = 0
        grad = 0
        for i in range(self.n_splits):
            vali, gradi = self.dict_crits[i].get_val_grad(
                self.models[i], log_alpha, get_beta_jac_v, max_iter=max_iter,
                tol=tol, compute_jac=compute_jac)
            val += vali
            if gradi is not None:
                grad += gradi
        val /= self.n_splits
        if gradi is not None:
            grad /= self.n_splits
        else:
            grad = None
        if monitor is not None:
            monitor(val, grad, log_alpha=log_alpha)
        return val, grad
