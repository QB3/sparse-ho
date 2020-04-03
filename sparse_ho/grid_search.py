import numpy as np
from numpy.linalg import norm
from sklearn.linear_model import Lasso

from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.utils import WarmStart


def grid_searchMCP(
        X, y, log_alphas, log_gammas, X_val, y_val, X_test, y_test,
        tol, monitor, maxit=1000, sk=False, criterion="cv", sigma=1,
        beta_star=None, random=False, max_evals=50, t_max=10000):

    for log_gamma in log_gammas:
        list_ = [[log_alpha, log_gamma] for log_alpha in log_alphas]
        grid_searchCV(
            X, y, list_, X_val, y_val, X_test, y_test,
            tol, monitor, maxit=maxit, sk=sk, criterion=criterion,
            sigma=sigma, beta_star=beta_star, random=random,
            max_evals=max_evals, model="mcp")
        if monitor.times[-1] > t_max:
            break


def grid_searchCV(
        X, y, log_alphas, X_val, y_val, X_test, y_test,
        tol, monitor, maxit=1000, sk=False, criterion="cv", sigma=1,
        beta_star=None, random=False, max_evals=50, model="lasso"):
    mask = None
    dense = None
    warm_start = WarmStart()
    if random:
        log_alphas = np.random.uniform(
            np.min(log_alphas), np.max(log_alphas), size=50)
        log_alphas = -np.sort(-log_alphas)
    for log_alpha in log_alphas:
        mask, dense = get_val_grid(
            X, y, log_alpha, X_val, y_val, X_test, y_test, tol, monitor,
            warm_start, sk=sk, maxit=maxit, sigma=sigma,
            criterion=criterion, beta_star=beta_star)


def get_val_grid(
        X, y, log_alpha, X_val, y_val, X_test, y_test,
        tol, monitor, warm_start, mask0=None, dense0=None, maxit=1000,
        sk=False, criterion="cv", random_state=42,
        C=2.0, gamma_sure=0.3, sigma=1, beta_star=None):
    alpha = np.exp(log_alpha)
    n_samples, n_features = X.shape

    mask0, dense0, mask20, dense20 = (
        warm_start.mask_old, warm_start.beta_old, warm_start.mask_old2,
        warm_start.beta_old2)

    if criterion == "cv":
        mask2, dense2, rmse = None, None, None
        if sk:
            clf = Lasso(
                alpha=alpha, fit_intercept=False, warm_start=True, tol=tol)
            clf.fit(X, y)
            coef_ = clf.coef_
            mask = coef_ != 0
            dense = coef_[mask]
        else:
            mask, dense = get_beta_jac_iterdiff(
                X, y, log_alpha, mask0=mask0, dense0=dense0, maxit=maxit,
                tol=tol, compute_jac=False)
    elif criterion == "sure":
        val_test = 0
        epsilon = C * sigma / (n_samples) ** gamma_sure
        rng = np.random.RandomState(random_state)
        delta = rng.randn(n_samples)  # sample random noise for MCMC step
        y2 = y + epsilon * delta

        mask, dense = get_beta_jac_iterdiff(
            X, y, log_alpha, mask0=mask0, dense0=dense0, maxit=maxit,
            tol=tol, compute_jac=False)
        mask2, dense2 = get_beta_jac_iterdiff(
            X, y2, log_alpha, mask0=mask20, dense0=dense20, maxit=maxit,
            tol=tol, compute_jac=False)

    if criterion == "cv":
        val = norm(y_val - X_val[:, mask] @ dense) ** 2 / X_val.shape[0]
        val_test = norm(
            y_test - X_test[:, mask] @ dense) ** 2 / X_test.shape[0]
    elif criterion == "sure":
        val_test = 0
        dof = (X[:, mask2] @ dense2 - X[:, mask] @ dense) @ delta
        dof /= epsilon
        val = norm(y - X[:, mask] @ dense) ** 2
        val -= n_samples * sigma ** 2
        val += 2 * sigma ** 2 * dof

    warm_start(mask, dense, None, mask2, dense2, None)

    if beta_star is not None:
        diff_beta = beta_star.copy()
        diff_beta[mask] -= dense
        rmse = norm(diff_beta)

    monitor(val, val_test, log_alpha.copy(), rmse=rmse)

    return mask, dense
