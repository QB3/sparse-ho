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
import matplotlib.pyplot as plt
import pandas as pd
from numpy.linalg import norm

from celer import Lasso, LassoCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from scipy.linalg import toeplitz

from sparse_ho.models import wLasso
from sparse_ho.criterion import CV
from sparse_ho.implicit_forward import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search

##############################################################################
# Dataset creation
##############################################################################
# X, y = make_regression(n_samples=600, n_features=600, noise=60, random_state=2)
n_samples = 600
n_features = 600
rng = check_random_state(0)
X = rng.multivariate_normal(
    size=n_samples, mean=np.zeros(n_features),
    cov=toeplitz(0.5 ** np.arange(n_features)))


# Create true regression coefficients of 5 non-zero values
w_true = np.zeros(n_features)
size_supp = 5
idx = rng.choice(
    X.shape[0], size_supp, replace=False)
w_true[idx] = (-1) ** np.arange(size_supp)
noise = rng.randn(n_samples)
y = X @ w_true
y += noise / norm(noise) * 0.5 * norm(y)

# Here we split the dataset (X, y) in 3:
# the regression coefficients will be determined using X_train, y_train
# the regularization parameter will be calibrated using X_val, y_val
# the model is then tested on unseen data X_test, y_test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.3, random_state=2)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.5, random_state=2)
##############################################################################


# max penalty value
alpha_max = np.max(np.abs(X_train.T.dot(y_train))) / X_train.shape[0]
n_alphas = 30  # number of iter in the line search, ie 30 evals of the gradient
alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)


##############################################################################
# Vanilla LassoCV
print("========== Celer's LassoCV started ===============")
model_cv = LassoCV(
    verbose=False, fit_intercept=False, alphas=alphas, tol=1e-7, max_iter=100,
    cv=2).fit(X_train_val, y_train_val)

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
lasso_sho = Lasso(fit_intercept=False, max_iter=10, warm_start=True)

model_sho = wLasso(X_train, y_train, estimator=lasso_sho)
criterion_sho = CV(X_val, y_val, model_sho, X_test=X_test, y_test=y_test)
algo_sho = ImplicitForward(criterion_sho)
monitor = Monitor()
grad_search(
    algo_sho, alpha0, monitor, n_outer=20, tol=1e-6)


# MSE on validation set
mse_sho_val = mean_squared_error(y_val, lasso_sho.predict(X_val))

# MSE on test set, ie unseen data
mse_sho_test = mean_squared_error(y_test, lasso_sho.predict(X_test))


print("Sparse-ho: Mean-squared error on validation data %f" % mse_sho_val)
print("Sparse-ho: Mean-squared error on test (unseen) data %f" % mse_sho_test)


labels = ['wLasso val', 'wLasso test', 'Lasso CV']

df = pd.DataFrame(
    np.array([mse_sho_val, mse_sho_test, mse_cv]).reshape((1, -1)),
    columns=labels)
df.plot.bar(rot=0)
plt.xlabel("Estimator")
plt.ylabel("Mean square error")
plt.tight_layout()
plt.show()
