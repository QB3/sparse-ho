import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sparse_ho.utils_cross_entropy import (
    cross_entropy, grad_cross_entropy, accuracy)


class LogisticMulticlass():
    """Multiclass logistic loss.

    Parameters
    ----------
    idx_train: ndarray
        indices of the training set
    idx_val: ndarray
        indices of the validation set
    algo: instance of ``sparse_ho.base.AlgoModel``
        A model that follows the sparse_ho API.
    idx_test: ndarray
        indices of the test set

    Attributes
    ----------
    dict_models: dict
        dict with the models corresponding to each class.
    """

    def __init__(self, idx_train, idx_val, algo, idx_test=None):
        self.idx_train = idx_train
        self.idx_val = idx_val
        # passing test is dirty but we need it for the multiclass logreg
        self.idx_test = idx_test
        # passing algo is dirty but we need it for the multiclass logreg
        self.algo = algo
        self.dict_models = None

    def _initialize(self, model, X, y):
        enc = OneHotEncoder(sparse=False)  # maybe remove the sparse=False
        # split data set in test validation and train
        self.one_hot_code = enc.fit_transform(pd.DataFrame(y))

        self.n_classes = self.one_hot_code.shape[1]

        # dict with all the one vs all models
        self.dict_models = {}
        for k in range(self.n_classes):
            self.dict_models[k] = copy.deepcopy(model)
        self.dict_warm_start = {}
        self.n_samples, self.n_features = X.shape

    def get_val_grad(
            self, model, X, y, log_alpha, compute_beta_grad, monitor,
            tol=1e-3):
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
        monitor: instance of Monitor.
            Monitor.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        """
        # TODO use sparse matrices
        if self.dict_models is None:
            self._initialize(model, X, y)
        all_betas = np.zeros((self.n_features, self.n_classes))
        all_jacs = np.zeros((self.n_features, self.n_classes))
        for k in range(self.n_classes):
            mask0, dense0, jac0 = self.dict_warm_start.get(
                k, (None, None, None))
            mask, dense, jac = self.algo.get_beta_jac(
                X[self.idx_train, :], self.one_hot_code[self.idx_train, k],
                log_alpha[k], self.dict_models[k], None, mask0=mask0,
                dense0=dense0,
                quantity_to_warm_start=jac0, tol=tol)
            self.dict_warm_start[k] = (mask, dense, jac)
            all_betas[mask, k] = dense  # maybe use np.ix_
            all_jacs[mask, k] = jac  # maybe use np.ix_
        acc_val = accuracy(
            all_betas, X[self.idx_val, :], self.one_hot_code[self.idx_val, :])
        val = cross_entropy(
            all_betas, X[self.idx_val, :], self.one_hot_code[self.idx_val, :])
        grad = self.grad_total_loss(
            all_betas, all_jacs, X[self.idx_val, :],
            self.one_hot_code[self.idx_val, :])

        if self.idx_test is not None:
            acc_test = accuracy(
                all_betas, X[self.idx_test, :], self.one_hot_code[
                    self.idx_test, :])
            print(
                "Value outer %f || Acc. validation %f || Acc. test %f" %
                (val, acc_val, acc_test))
        else:
            acc_test = None
            print("Value outer %f || Acc. validation %f" %
                  (val, acc_val))

        monitor(
            val, alpha=np.exp(log_alpha), grad=grad.copy(), acc_val=acc_val,
            acc_test=acc_test)

        self.all_betas = all_betas
        return val, grad

    def get_val(
            self, model, X, y, log_alpha, compute_beta_grad, monitor,
            tol=1e-3):
        # TODO not the same as for other losses?
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
        compute_beta_grad: callable
            Returns the regression coefficients beta and the hypergradient.
        monitor: instance of Monitor.
            Monitor.
        tol: float, optional (default=1e-3)
            Tolerance for the inner problem.
        """
        if self.dict_models is None:
            self._initialize(model, X, y)
        all_betas = np.zeros((self.n_features, self.n_classes))
        for k in range(self.n_classes):
            mask0, dense0, jac0 = self.dict_warm_start.get(
                k, (None, None, None))
            mask, dense, jac = self.algo.get_beta_jac(
                X[self.idx_train, :], self.one_hot_code[self.idx_train, k],
                log_alpha[k], self.dict_models[k], None, mask0=mask0,
                dense0=dense0,
                quantity_to_warm_start=jac0, tol=tol)
            self.dict_warm_start[k] = (mask, dense, jac)
            all_betas[mask, k] = dense  # maybe use np.ix_
        acc_val = accuracy(
            all_betas, X[self.idx_val, :], self.one_hot_code[self.idx_val, :])
        acc_test = accuracy(
            all_betas, X[self.idx_test, :],
            self.one_hot_code[self.idx_test, :])
        val = cross_entropy(
            all_betas, X[self.idx_val, :], self.one_hot_code[self.idx_val, :])
        monitor(
            val, alpha=np.exp(log_alpha), grad=None, acc_val=acc_val,
            acc_test=acc_test)
        print("Value outer %f || Accuracy validation %f || Accuracy test %f" %
              (val, acc_val, acc_test))
        self.all_betas = all_betas
        return val

    def proj_hyperparam(self, model, X, y, log_alpha):
        """Project hyperparameter on admissible range of values

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
        """
        # TODO doesn't an other object do this?
        # TODO model not needed I think
        log_alpha_max = model.compute_alpha_max(X, y)
        log_alpha[log_alpha < log_alpha_max - 7] = log_alpha_max - 7
        log_alpha[log_alpha > log_alpha_max - np.log(0.9)] = (
            log_alpha_max - np.log(0.9))
        return log_alpha

    def grad_total_loss(self, all_betas, all_jacs, X, Y):
        """Compute the gradient of the multiclass logistic loss.

        Parameters
        ----------
        all_betas: array-like, shape (n_features, n_classes)
            Solutions of the optimization problems corresponding to each class.
        all_jacs: array-like, shape (n_features, n_classes)
            Jacobians of the optimization problems corresponding to each class.
        X: array-like, shape (n_samples, n_features)
            Design matrix.
        Y: ndarray, shape (n_samples, n_classes)
            One hot encoding representation of the observation y.
        """
        grad_ce = grad_cross_entropy(all_betas, X, Y)
        grad_total = (grad_ce * all_jacs).sum(axis=0)
        return grad_total

    # def grad_k_loss(self, all_betas, jack, X, Y, k):
    #     grad_ce = grad_cross_entropyk(all_betas, X, Y, k)
    #     grad_k = grad_ce @ jack
    #     return grad_k
