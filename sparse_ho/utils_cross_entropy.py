import numpy as np
import sklearn
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


def accuracy(betas, X, Y):
    scores = X @ betas
    idx_max = np.argmax(scores, axis=1)
    idx_true = np.argmax(Y, axis=1)  # TODO to improve
    # acc = (idx_max == idx_true).mean()
    acc = sklearn.metrics.accuracy_score(idx_max, idx_true)
    return acc


def grad_cross_entropy(betas, X, Y):
    """Compute gradient of cross-entropy wrt betas
    betas: array of size (n_features, n_classes)
    X: {ndarray, sparse matrix} of (n_samples, n_features)
    Y: {ndarray, sparse matrix} of (n_samples, n_classes)
    """
    n_samples = X.shape[0]
    sm = softmax(X @ betas)
    weights = sm - Y
    grad = (X.T @ weights) / n_samples

    return grad


# def grad_cross_entropyk(betas, X, Y, k):
#     """Compute gradient of cross-entropy wrt betas
#     betas: array of size (n_features, n_classes)
#     X: {ndarray, sparse matrix} of (n_samples, n_features)
#     Y: {ndarray, sparse matrix} of (n_samples, n_classes)
#     """
#     n_samples = X.shape[0]
#     sm = softmax(X @ betas)
#     weights = sm[:, k] - Y[:, k]
#     gradk = (X.T @ weights) / n_samples

#     return gradk
