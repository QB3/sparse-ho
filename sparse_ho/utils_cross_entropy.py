import numpy as np
from scipy.special import logsumexp


def softmax(Xbetas):
    # Stable softmax
    exp = np.exp(Xbetas - np.max(Xbetas, axis=1, keepdims=True))
    norms = np.sum(exp, axis=1)[:, np.newaxis]
    return exp / norms


def log_softmax(Xbetas):
    # Stable log softmax
    return Xbetas - logsumexp(Xbetas, axis=1, keepdims=True)


def cross_entropy(betas, X, Y):
    """cross-entropy"""
    n_samples, n_features = X.shape
    result = - np.sum(log_softmax(X @ betas) * Y) / n_samples
    if np.isnan(result):
        import ipdb
        ipdb.set_trace()
    return result


def grad_cross_entropy(betas, X, Y):
    """Compute gradient of cross-entropy wrt betas
    betas: array of size (n_features, n_classes)
    X: {ndarray, sparse matrix} of (n_samples, n_features)
    Y: {ndarray, sparse matrix} of (n_samples, n_classes)
    """
    n_samples = X.shape[0]
    n_classes = Y.shape[1]
    sm = softmax(X @ betas)

    grad = np.empty_like(betas)
    for k in range(n_classes):
        weights = sm[:, k] * Y.sum(axis=1) - Y[:, k]
        grad[:, k] = (X.T @ weights) / n_samples

    return grad
