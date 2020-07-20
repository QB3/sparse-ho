import numpy as np
from sklearn.linear_model import ElasticNet as ElasticNetsk
from sparse_ho.models import ElasticNet as Elastic
from sparse_ho.forward import get_beta_jac_iterdiff
from sklearn.datasets import make_regression
# from sparse_ho.implicit_forward import get_beta_jac_fast_iterdiff
# from sparse_ho.criterion import CV
# from sparse_ho.forward import Forward
# from sparse_ho.implicit_forward import ImplicitForward
# from sparse_ho.ho import grad_search
# from sparse_ho.utils import Monitor


n_samples = 100
n_features = 100
n_active = 100
tol = 1e-16
alpha_1 = 0.01
alpha_2 = 0.2
log_alpha1 = np.log(alpha_1)
log_alpha2 = np.log(alpha_2)

max_iter = 50000
X_train, y_train, beta_star = make_regression(
    shuffle=False, random_state=15, n_samples=n_samples, n_features=n_features,
    n_informative=n_features, n_targets=1, coef=True)

X_val, y_val, beta_star = make_regression(
    shuffle=False, random_state=125, n_samples=n_samples, n_features=n_features,
    n_informative=n_features, n_targets=1, coef=True)

model = Elastic(X_train, y_train, log_alpha1, log_alpha2, max_iter=max_iter, tol=tol)


def get_v(mask, dense):
    return 2 * (X_val[:, mask].T @ (
        X_val[:, mask] @ dense - y_val)) / X_val.shape[0]


def test_beta_jac():
    supp1, dense1, jac1 = get_beta_jac_iterdiff(
        X_train, y_train, np.array([log_alpha1, log_alpha2]), tol=tol,
        model=model, compute_jac=True, max_iter=max_iter)

    clf = ElasticNetsk(
        alpha=(alpha_1 + alpha_2), fit_intercept=False,
        l1_ratio=alpha_1 / (alpha_1 + alpha_2),
        tol=1e-12, max_iter=max_iter)
    clf.fit(X_train, y_train)
    import ipdb; ipdb.set_trace()

    assert np.allclose(dense1, clf.coef_[clf.coef_ != 0])


if __name__ == '__main__':
    test_beta_jac()
