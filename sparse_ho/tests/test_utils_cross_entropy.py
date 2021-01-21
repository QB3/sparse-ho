import numpy as np

from scipy.optimize import check_grad
from scipy.sparse import csc_matrix
from sklearn.preprocessing import OneHotEncoder

from sparse_ho.utils_cross_entropy import cross_entropy, grad_cross_entropy


if __name__ == '__main__':
    rng = np.random.RandomState(42)
    K = 5
    X = rng.randn(120, 100)
    X = csc_matrix(X)
    y = rng.choice(range(K), size=X.shape[0])
    Y = OneHotEncoder().fit_transform(y[:, None]).toarray()
    betas = rng.randn(X.shape[1], K)

    def f(x):
        return cross_entropy(x.reshape(X.shape[1], K), X, Y)

    def gradf(x):
        return grad_cross_entropy(x.reshape(X.shape[1], K), X, Y).ravel()

    np.testing.assert_allclose(
        check_grad(f, gradf, x0=betas.ravel()), 0, atol=1e-5)
