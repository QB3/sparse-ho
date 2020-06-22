import numpy as np
from sklearn.model_selection import train_test_split
from scipy.sparse import issparse

from sparse_ho.ho import _grad_search


def grad_search_CV(
        X, y, Model, Criterion, Algo, log_alpha0, monitor, n_outer=100,
        verbose=True, cv=5, random_state=0, test_size=0.33,
        tolerance_decrease='constant', tol=1e-5,
        t_max=10000, beta_star=None,):

    dict_algo = {}

    if isinstance(cv, int):
        for i in range(cv):
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=test_size, random_state=cv)

            if issparse(X_train):
                X_train = X_train.tocsc().copy()
            if issparse(X_val):
                X_val = X_val.tocsc().copy()

            model = Model(X_train, y_train, log_alpha0)

            criterion = Criterion(
                X_val, y_val, model, X_test=X_val, y_test=y_val)

            algo = Algo(criterion, tol_jac=1e-3, n_iter_jac=100, use_sk=True)
            dict_algo[i] = algo
        n_splits = cv
    else:
        for i, (train, val) in enumerate(cv.split(X)):
            # import ipdb; ipdb.set_trace()
            X_train = X[train, :]
            y_train = y[train]
            X_val = X[val, :]
            y_val = y[val]

            if issparse(X_train):
                X_train = X_train.tocsc().copy()
            if issparse(X_val):
                X_val = X_val.tocsc().copy()

            model = Model(X_train, y_train, log_alpha0)

            criterion = Criterion(
                X_val, y_val, model, X_test=X_val, y_test=y_val)

            algo = Algo(criterion, tol_jac=1e-3, n_iter_jac=100, use_sk=True)
            dict_algo[i] = algo
        n_splits = cv.n_splits

    def _get_val_grad(lambdak, tol):
        val = 0
        grad = np.zeros_like(log_alpha0)
        for i in range(n_splits):
            val_i, grad_i = algo.get_val_grad(
                lambdak, tol=algo.criterion.model.tol,
                beta_star=beta_star)
            val += val_i
            grad += grad_i
        val /= n_splits
        grad /= n_splits
        return val, grad

    def _proj_param(lambdak):
        return dict_algo[0].criterion.model.proj_param(lambdak)

    return _grad_search(
        _get_val_grad, _proj_param, log_alpha0, monitor,
        dict_algo[0], n_outer=n_outer,
        verbose=verbose, tolerance_decrease=tolerance_decrease, tol=tol,
        t_max=t_max)
