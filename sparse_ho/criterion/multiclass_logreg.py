import copy
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sparse_ho.utils_cross_entropy import (
    cross_entropy, grad_cross_entropy, grad_cross_entropyk, accuracy)


class LogisticMulticlass():
    """Multiclass logistic loss.
    """
    def __init__(self, idx_val, idx_train, algo):
        """
        Parameters
        ----------
        idx_train: np.array
            indices of the training set
        idx_test: np.array
            indices of the testing set
        """
        self.idx_train = idx_train
        self.idx_val = idx_val
        # passing algo is dirty but we need it for the multiclass logreg
        self.algo = algo
        self.dict_models = None

    def _initialize(self, model, X, y):
        enc = OneHotEncoder(sparse=False)  # maybe remove the sparse=False
        # split data set in test validation and train
        self.one_hot_code = enc.fit_transform(pd.DataFrame(y))
        # self.one_hot_code_train = self.one_hot_code[idx_train, :]
        # self.one_hot_code_val = self.one_hot_code[idx_val, :]
        # self.one_hot_code_test = self.one_hot_code[idx_test, :]

        self.n_classes = self.one_hot_code.shape[1]

        # dict with all the one vs all models
        self.dict_models = {}
        for k in range(self.n_classes):
            self.dict_models[k] = copy.deepcopy(model)
        self.dict_warm_start = {}
        self.n_samples, self.n_features = X.shape
        # self.all_betas = np.zeros((self.n_features, self.n_classes))

    def get_val_grad(
            self, model, X, y, log_alpha, get_beta_jac_v, monitor, tol=1e-3):
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
                quantity_to_warm_start=jac0, compute_jac=True, tol=tol)
            self.dict_warm_start[k] = (mask, dense, jac)
            all_betas[mask, k] = dense  # maybe use np.ix_
            all_jacs[mask, k] = jac  # maybe use np.ix_
        acc_val = accuracy(
            all_betas, X[self.idx_val, :], self.one_hot_code[self.idx_val, :])
        # acc_test = accuracy(
        #     all_betas, self.X_test, self.one_hot_code[self.idx_test])
        # TODO use a callback function
        val = cross_entropy(
            all_betas, X[self.idx_val, :], self.one_hot_code[self.idx_val, :])
        grad = self.grad_total_loss(
            all_betas, all_jacs, X[self.idx_val, :],
            self.one_hot_code[self.idx_val, :])
        monitor(
            val, log_alpha=log_alpha.copy(), grad=grad.copy(), acc_val=acc_val)
        self.all_betas = all_betas
        return val, grad

    def get_val(self, log_alpha, tol=1e-3):
        # TODO use sparse matrices
        # TODO add warm start
        all_betas = np.zeros((self.n_features, self.n_classes))
        for k in range(self.n_classes):
            model = self.dict_models[k]
            # TODO add warm start
            mask0, dense0, jac0 = self.dict_warm_start.get(
                k, (None, None, None))
            mask, dense, jac = self.algo.get_beta_jac(
                model.X, model.y, log_alpha[k], model, None, mask0=None,
                dense0=None, quantity_to_warm_start=None, compute_jac=True)
            self.dict_warm_start[k] = (mask, dense, jac)
            all_betas[mask, k] = dense  # maybe use np.ix_
        val = cross_entropy(all_betas, self.X_val, self.one_hot_code_val)
        # acc_val = accuracy(all_betas, self.X_val, self.one_hot_code_val)
        # acc_test = accuracy(all_betas, self.X_test, self.one_hot_code_test)
        # monitor(
        #     val, log_alpha=log_alpha.copy(), grad=None, acc_val=acc_val,
        #     acc_test=acc_test)
        return val

    def get_val_monitor(self, log_alpha, monitor, tol=1e-3):
        # TODO use sparse matrices
        # TODO add warm start
        all_betas = np.zeros((self.n_features, self.n_classes))
        for k in range(self.n_classes):
            model = self.dict_models[k]
            # TODO add warm start
            mask0, dense0, jac0 = self.dict_warm_start.get(
                k, (None, None, None))
            mask, dense, jac = self.algo.get_beta_jac(
                model.X, model.y, log_alpha[k], model, None, mask0=None,
                dense0=None, quantity_to_warm_start=None, compute_jac=True)
            self.dict_warm_start[k] = (mask, dense, jac)
            all_betas[mask, k] = dense  # maybe use np.ix_
        self.all_betas = all_betas
        val = cross_entropy(self.all_betas, self.X_val, self.one_hot_code_val)
        acc_val = accuracy(self.all_betas, self.X_val, self.one_hot_code_val)
        acc_test = accuracy(
            self.all_betas, self.X_test, self.one_hot_code_test)
        monitor(
            val, log_alpha=log_alpha.copy(), grad=None, acc_val=acc_val,
            acc_test=acc_test)
        return val

    def proj_hyperparam(self, model, X, y, log_alpha):
        log_alpha_max = model.compute_alpha_max(X, y)
        log_alpha[log_alpha < log_alpha_max - 7] = log_alpha_max - 7
        log_alpha[log_alpha > log_alpha_max - np.log(0.9)] = (
            log_alpha_max - np.log(0.9))
        return log_alpha

    def grad_total_loss(self, all_betas, all_jacs, X, Y):
        grad_ce = grad_cross_entropy(all_betas, X, Y)
        grad_total = (grad_ce * all_jacs).sum(axis=0)
        return grad_total

    def grad_k_loss(self, all_betas, jack, X, Y, k):
        grad_ce = grad_cross_entropyk(all_betas, X, Y, k)
        grad_k = grad_ce @ jack
        return grad_k
