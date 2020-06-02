"""
Method comparison on Lasso
==========================

The aim of this example is to demonstrate on a simple
dateset how methods compare.

"""


import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets
from scipy.sparse import csc_matrix

from sparse_ho.ho import grad_search
from sparse_ho.utils import Monitor
from sparse_ho.models import SparseLogreg
from sparse_ho.criterion import Logistic
from sparse_ho.implicit_forward import ImplicitForward
# from sparse_ho.grid_search import grid_searchCV
# from sparse_ho.bayesian import hyperopt_lasso

from sparse_ho.datasets.real import get_rcv1
from expes.utils import configure_plt


# X_train, X_val, X_test, y_train, y_val, y_test = get_rcv1()
# y_train[y_train == -1.0] = 0.0
# y_val[y_val == -1.0] = 0.0
# y_test[y_test == -1.0] = 0.0

n_samples = 100
n_features = 1000
X_train, y_train = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features, n_informative=50,
        random_state=10, flip_y=0.1, n_redundant=0)
X_train_s = csc_matrix(X_train)


X_val, y_val = datasets.make_classification(
        n_samples=n_samples,
        n_features=n_features, n_informative=50,
        random_state=12, flip_y=0.1, n_redundant=0)

X_val_s = csc_matrix(X_val)


n_samples, n_features = X_train.shape

alpha_max = 1 / 4
alpha_max = np.abs((y_train - np.mean(y_train) * (1 - np.mean(y_train))).T @ X_train).max() / n_samples
maxit = 1000

log_alpha0 = np.log(0.9 * alpha_max)
tol = 1e-12

model = SparseLogreg(X_train, y_train, log_alpha0, max_iter=10000, tol=tol)
criterion = Logistic(X_val, y_val, model)
monitor = Monitor()
algo = ImplicitForward(criterion, tol_jac=tol, n_iter_jac=5000)
grad_search(algo, log_alpha0, monitor, n_outer=50, tol=tol)
