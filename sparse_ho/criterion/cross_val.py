import copy
import numpy as np
from sklearn.model_selection import check_cv
from sparse_ho.criterion.base import BaseCriterion


class CrossVal(BaseCriterion):
    """Cross-validation loss.

    Parameters
    ----------
    criterion : instance of ``BaseCriterion``
        A criterion that follows the sparse-ho API.
    cv : int, cross-validation generator or iterable, default=None
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 5-fold cross-validation,
        - int, to specify the number of folds.
        - scikit-learn CV splitter
        - An iterable yielding (train, test) splits as arrays of indices.

        For int/None inputs, KFold is used.

    Attributes
    ----------
    dict_crits : dict
        The instances of criterion used for each fold.
    """

    # XXX TODO pass criterion as a string, MSE, logistic
    # do directly crossval in MSE and Logistic

    def __init__(self, criterion, cv=None):
        self.criterion = criterion
        self.cv = check_cv(cv)
        self.dict_crits = None
        self.dict_models = None
        self.n_splits = None

    def _initialize(self, model, X):
        self.dict_crits = {}
        self.dict_models = {}
        self.n_splits = self.cv.get_n_splits(X)

        for i, (idx_train, idx_val) in enumerate(self.cv.split(X)):
            self.dict_crits[i] = copy.deepcopy(self.criterion)
            self.dict_crits[i].idx_train = idx_train
            self.dict_crits[i].idx_val = idx_val
            self.dict_models[i] = copy.deepcopy(model)

    def get_val(self, model, X, y, log_alpha, monitor=None, tol=1e-3):
        """Get value of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        monitor: instance of Monitor.
            Monitor.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        """

        if self.dict_crits is None:
            self._initialize(model, X)
        val = np.mean([
            self.dict_crits[i].get_val(
                self.dict_models[i], X, y, log_alpha, tol=tol) for i in range(
                    self.n_splits)])
        monitor(val, None, alpha=np.exp(log_alpha))
        return val

    def get_val_grad(
            self, model, X, y, log_alpha, compute_beta_grad, max_iter=10000,
            tol=1e-5, monitor=None):
        """Get value and gradient of criterion.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float or np.array
            Logarithm of hyperparameter.
        compute_beta_grad: callable
            Returns the regression coefficients beta and the hypergradient.
        max_iter: int
            Maximum iteration for the inner optimization problem.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        monitor: instance of Monitor.
            Monitor.
        """
        if self.dict_crits is None:
            self._initialize(model, X)

        val = 0
        grad = 0
        for i in range(self.n_splits):
            vali, gradi = self.dict_crits[i].get_val_grad(
                self.dict_models[i], X, y, log_alpha, compute_beta_grad,
                max_iter=max_iter, tol=tol)
            val += vali
            if gradi is not None:
                grad += gradi
        val /= self.n_splits
        if gradi is not None:
            grad /= self.n_splits
        else:
            grad = None
        if monitor is not None:
            monitor(val, grad, alpha=np.exp(log_alpha))
        return val, grad

    def get_val_outer(cls, *args, **kwargs):
        """Get value of outer criterion.

        This is not implemented because for CV the loss is computed on each
        fold.

        Parameters
        ----------
        cls: TODO
        """
        return NotImplemented

    def proj_hyperparam(self, model, X, y, log_alpha):
        """Project hyperparameter on a range of admissible values.

        Parameters
        ----------
        model: instance of ``sparse_ho.base.BaseModel``
            A model that follows the sparse_ho API.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        y: ndarray, shape (n_samples,)
            Observation vector.
        log_alpha: float
            Logarithm of hyperparameter.
        """
        # TODO is this not done by the models?
        # TODO to improve this proj_hyperparam procedure
        return model.proj_hyperparam(X, y, log_alpha)
