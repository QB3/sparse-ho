import numpy as np
# from sklearn.svm import LinearSVR
from sparse_ho.models import SVR
# from sparse_ho.forward import get_beta_jac_iterdiff
from sklearn.datasets import make_regression
n_samples = 100
n_features = 100
n_active = 100
SNR = 5.0
tol = 1e-12
C = 0.1
log_C = np.log(C)
epsilon = 0.05
log_epsilon = np.log(epsilon)

X_train, y_train, beta_star = make_regression(
    shuffle=False, random_state=10, n_samples=n_samples, n_features=n_features,
    n_informative=n_features, n_targets=1, coef=True)

model = SVR(X_train, y_train, log_C, log_epsilon, max_iter=10000, tol=tol)
