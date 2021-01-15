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
from scipy.linalg import toeplitz

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.utils import check_random_state
from celer import Lasso, LassoCV

from sparse_ho.models import WeightedLasso
from sparse_ho.criterion import HeldOutMSE
from sparse_ho import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search
from sparse_ho.optimizers import LineSearch


##############################################################################
# Dataset creation
n_samples = 900
n_features = 600
rng = check_random_state(0)
X = rng.multivariate_normal(
    size=n_samples, mean=np.zeros(n_features),
    cov=toeplitz(0.5 ** np.arange(n_features)))


# Create true regression coefficients of 5 non-zero values
w_true = np.zeros(n_features)
size_supp = 5
idx = rng.choice(X.shape[1], size_supp, replace=False)
w_true[idx] = (-1) ** np.arange(size_supp)
noise = rng.randn(n_samples)
y = X @ w_true
y += noise / norm(noise) * 0.5 * norm(y)
##############################################################################
X, X_test, y, y_test = train_test_split(X, y, test_size=0.333, random_state=0)

n_samples = X.shape[0]
idx_train = np.arange(0, n_samples // 2)
idx_val = np.arange(n_samples // 2, n_samples)
##############################################################################

##############################################################################
# Max penalty value
alpha_max = np.max(np.abs(X[idx_train, :].T.dot(y[idx_train])))
alpha_max /= len(idx_train)
n_alphas = 30
alphas = alpha_max * np.geomspace(1, 0.001, n_alphas)
##############################################################################

##############################################################################
# Vanilla LassoCV
print("========== Celer's LassoCV started ===============")
model_cv = LassoCV(
    verbose=False, fit_intercept=False, alphas=alphas, tol=1e-7, max_iter=100,
    cv=2, n_jobs=2).fit(X, y)

# Measure mse on test
mse_cv = mean_squared_error(y_test, model_cv.predict(X_test))
print("Vanilla LassoCV: Mean-squared error on test data %f" % mse_cv)
##############################################################################


##############################################################################
# Weighted Lasso with sparse-ho.
# We use the vanilla lassoCV coefficients as a starting point
log_alpha0 = np.log(model_cv.alpha_) * np.ones(n_features)
# Weighted Lasso: Sparse-ho: 1 param per feature
estimator = Lasso(fit_intercept=False, max_iter=10, warm_start=True)
model = WeightedLasso(estimator=estimator)
criterion = HeldOutMSE(idx_train, idx_val)
algo = ImplicitForward()
monitor = Monitor()
optimizer = LineSearch(n_outer=20, tol=1e-6, verbose=True)
results = grad_search(
    algo, criterion, model, optimizer, X, y, log_alpha0, monitor)
##############################################################################

##############################################################################
# MSE on validation set
mse_sho_val = mean_squared_error(y[idx_val], estimator.predict(X[idx_val, :]))

# MSE on test set, ie unseen data
mse_sho_test = mean_squared_error(y_test, estimator.predict(X_test))


print("Sparse-ho: Mean-squared error on validation data %f" % mse_sho_val)
print("Sparse-ho: Mean-squared error on test (unseen) data %f" % mse_sho_test)


labels = ['WeightedLasso val', 'WeightedLasso test', 'Lasso CV']

df = pd.DataFrame(
    np.array([mse_sho_val, mse_sho_test, mse_cv]).reshape((1, -1)),
    columns=labels)
df.plot.bar(rot=0)
plt.xlabel("Estimator")
plt.ylabel("Mean square error")
plt.tight_layout()
plt.show()
##############################################################################
