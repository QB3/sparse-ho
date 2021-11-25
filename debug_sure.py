

from sparse_ho.models import WeightedLasso
from sparse_ho.criterion import FiniteDiffMonteCarloSure
from sparse_ho import ImplicitForward
from sparse_ho.utils import Monitor
from sparse_ho.ho import grad_search
from sparse_ho.optimizers import GradientDescent
from celer.datasets import make_correlated_data
from celer import Lasso as celer_Lasso
from sklearn.model_selection import train_test_split
import numpy as np

X, y, w_true = make_correlated_data(
    n_samples=100, n_features=100, random_state=0, snr=5)

##############################################################################
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.333, random_state=0)

n_samples, n_features = X_train.shape
alpha_max_old = (np.abs(X_train.T @ y_train)).max() / n_samples
X_train /= alpha_max_old

alpha_max = (np.abs(X_train.T @ y_train)).max() / n_samples
alpha0 = 0.7 * alpha_max
# alpha0 = 0.7 * alpha_max
# log_alpha0 = np.log(alpha0)

tol = 1e-9
n_outer = 10

estimator = celer_Lasso(fit_intercept=False, max_iter=100, warm_start=True)

alpha0 = alpha0 * np.ones(n_features)
model = WeightedLasso(estimator=estimator)


sigma = 0.1
criterion = FiniteDiffMonteCarloSure(sigma=sigma)
algo = ImplicitForward(criterion)
optimizer = GradientDescent(
    n_outer=100, tol=1e-7, verbose=True, p_grad_norm=1.9)
monitor = Monitor()
grad_search(algo, criterion, model, optimizer,
            X_train, y_train, alpha0, monitor)
