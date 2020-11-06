"""
=====================================
Weighted Lasso with held-out test set
=====================================

This example shows how to perform hyperparameter optimization
for a weighted Lasso using a held-out validation set.
In particular we compare the weighted Lasso to LassoCV on a toy example
"""

# Authors: Quentin Bertrand <quentin.bertrand@inria.fr>
#          Quentin Klopfenstein <quentin.klopfenstein@u-bourgogne.fr>
#          Kenan Sehic
# License: BSD (3-clause)

import numpy as np

from celer import Lasso, LassoCV
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

from sparse_ho.models import wLasso
from sparse_ho.criterion import CV
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search

##############################################################################
# Dataset creation
##############################################################################
X, y = make_regression(n_samples=500, n_features=600, noise=0, random_state=2)

# Here we split the dataset (X, y) in 3:
# the regression coefficients will be determined using X_train, y_train
# the regularization parameter will be calibrated using X_val, y_val
# the model is then tested on unseen data X_test, y_test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.5)
##############################################################################


# max penalty value
alpha_max = np.max(np.abs(X_train.T.dot(y_train))) / X_train.shape[0]
n_alphas = 30  # number of iter in the line search, ie 30 evals of the gradient
alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)


##############################################################################
# vanilla LassoCV
print("========== Celer's LassoCV started ===============")
model_cv = LassoCV(
    verbose=False, fit_intercept=False, alphas=alphas, tol=1e-7, max_iter=100,
    cv=5).fit(X_train_val, y_train_val)

# measure mse on test
mse_cv = mean_squared_error(y_test, model_cv.predict(X_test))
print("Vanilla LassoCV: Mean-squared error on test data %f" % mse_cv)
##############################################################################


##############################################################################
# Weighted Lasso with sparse-ho
# We use the vanilla lassoCV coefficients as a starting point
alpha0 = np.log(model_cv.alpha_) * np.ones(X_train.shape[1])

##############################################################################
#  weighted Lasso: Sparse-ho: 1 param per feature
lasso_sho = Lasso(fit_intercept=False, max_iter=30, warm_start=True)
model_sho = wLasso(X_train, y_train, estimator=lasso_sho)
criterion_sho = CV(X_val, y_val, model_sho, X_test=X_test, y_test=y_test)
algo_sho = ImplicitForward(criterion_sho)
monitor = Monitor()
grad_search(
    algo_sho, alpha0, monitor, n_outer=20, tol=1e-6)


alphas = np.exp(monitor.log_alphas[-1])
lasso_sho.fit(X_train / alphas, y_train)
coef_ = lasso_sho.coef_ / alphas


# mse on validation set
mse_sho_val = mean_squared_error(y_val, X_val @ coef_)

# mse on test set, ie unseen data
mse_sho_test = mean_squared_error(y_test, X_test @ coef_)


print("Sparse-ho: Mean-squared error on validation data %f" % mse_sho_val)
print("Sparse-ho: Mean-squared error on test (unseen) data %f" % mse_sho_test)
