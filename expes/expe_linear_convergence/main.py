import numpy as np
from sparse_ho.datasets.synthetic import get_synt_data
from numpy.linalg import norm
from sklearn.linear_model import Lasso as Lasso_sk
from sklearn.linear_model import LogisticRegression
from scipy.sparse.linalg import cg
from sparse_ho.models import Lasso, SparseLogreg
import matplotlib.pyplot as plt
from sklearn import datasets
from sparse_ho.utils import sigma


def linear_cv(X, y, log_alpha, model, beta_star, jac_star,
              max_iter=1000, tol=1e-3, compute_jac=True):

    n_samples, n_features = X.shape
    is_sparse = False
    L = model.get_L(X, is_sparse=is_sparse)
    ############################################
    alpha = np.exp(log_alpha)
    try:
        alpha.shape[0]
        alphas = alpha.copy()
    except Exception:
        alphas = np.ones(n_features) * alpha
    ############################################
    # warm start for beta
    beta, r = model._init_beta_r(X, y, None, None)

    ############################################
    # warm start for dbeta
    dbeta, dr = model._init_dbeta_dr(
        X, y, mask0=None, dense0=None, jac0=None, compute_jac=compute_jac)
    # store the values of the objective function
    pobj0 = model._get_pobj(r, beta, alphas, y)
    pobj = []
    diff_beta = []
    diff_jac = []
    supp_id = []
    mask_star = beta_star != 0

    for i in range(max_iter):
        print("%i -st iteration over %i" % (i, max_iter))

        model._update_beta_jac_bcd(
            X, y, beta, dbeta, r, dr, alphas, L, compute_jac=compute_jac)

        pobj.append(model._get_pobj(r, beta, alphas, y))
        print(pobj[-1])

        diff_beta.append(norm(beta - beta_star))
        diff_jac.append(norm(dbeta - jac_star))
        mask = beta != 0
        if (mask == mask_star).all():
            supp_id.append(i)
        if i > 1:
            assert pobj[-1] - pobj[-2] <= 1e-5 * np.abs(pobj[0])
            print("relative decrease = ", (pobj[-2] - pobj[-1]) / pobj0)
        if (i > 1) and (pobj[-2] - pobj[-1] <= np.abs(pobj0 * tol)):
            break

    else:
        print('did not converge !')
        # raise RuntimeError('did not converge !')

    return diff_beta, diff_jac, i, supp_id


n_samples = 100
n_features = 1000
n_active = 5
SNR = 3.0
seed = 10
tol = 1e-16
max_iter = 10000

X, y, beta_star, noise, sigma_star = get_synt_data(
    "Gaussian", noise_type="Gaussian_iid", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=n_active, SNR=SNR,
    seed=seed)

alpha_max = norm(X.T @ y, ord=np.inf) / n_samples
alpha = 0.01 * alpha_max
clf = Lasso_sk(
    alpha=alpha, fit_intercept=False, warm_start=True,
    tol=tol, max_iter=10000)
clf.fit(X, y)
beta_star = clf.coef_
mask = beta_star != 0
dense = beta_star[mask]

v = - n_samples * alpha * np.sign(beta_star[mask])
mat_to_inv = X[:, mask].T  @ X[:, mask]

jac_temp = cg(mat_to_inv, v, atol=1e-10)
jac_star = np.zeros(n_features)
jac_star[mask] = jac_temp[0]

model = Lasso(X, y, np.log(alpha), max_iter=max_iter, tol=tol)
diff_beta, diff_jac, n_iter, supp_id = linear_cv(X, y, np.log(alpha), model, beta_star, jac_star,
                                                 max_iter=max_iter, tol=tol, compute_jac=True)


fig, axarr = plt.subplots(
    1, 2, sharex=False, sharey=False, figsize=[10, 4])
plt.figure()
axarr.flat[0].semilogy(range(n_iter+1), diff_beta, linewidth=2.0)
axarr.flat[1].semilogy(range(n_iter+1), diff_jac, linewidth=2.0)
axarr.flat[0].set_xlabel("epoch")
axarr.flat[0].set_ylabel("||beta - beta_star||")
axarr.flat[1].set_xlabel("epoch")
axarr.flat[1].set_ylabel("||jac - jac_Star||")
axarr.flat[0].axvline(x=supp_id[0], c='red', linestyle="--")
axarr.flat[1].axvline(x=supp_id[0], c='red', linestyle="--")
axarr.flat[0].set_title("Iterates convergence for the Lasso")
axarr.flat[1].set_title("Jacobian convergence for the Lasso")
plt.show()


# Same for the Logistic regression

n_samples = 100
n_features = 1000
SNR = 3.0
seed = 10
tol = 1e-14
max_iter = 10000

X_train, y_train = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features, n_informative=50,
    random_state=110, flip_y=0.1, n_redundant=0)

y_train[y_train == 0.0] = -1.0

X_val, y_val = datasets.make_classification(
    n_samples=n_samples,
    n_features=n_features, n_informative=50,
    random_state=122, flip_y=0.1, n_redundant=0)


alpha_max = norm(X_train.T @ y_train, ord=np.inf) / (2 * n_samples)
alpha = 0.05 * alpha_max

clf = LogisticRegression(penalty="l1", tol=1e-12, C=(
                         1 / (alpha * n_samples)), fit_intercept=False, max_iter=100000,
                         solver="liblinear")
clf.fit(X_train, y_train)

beta_star_logreg = clf.coef_
supp_sk = beta_star_logreg != 0
supp_sk = supp_sk[0, :].T
dense_sk = beta_star_logreg[0, supp_sk].T

v = - n_samples * alpha * np.sign(dense_sk)
r = y_train * (X_train[:, supp_sk] @ dense_sk)
mat_to_inv = X_train[:, supp_sk].T  @ np.diag(sigma(r) * (1 - sigma(r))) @ X_train[:, supp_sk]

jac_temp = cg(mat_to_inv, v, atol=1e-7)
jac_star = np.zeros(n_features)
jac_star[supp_sk] = jac_temp[0]

model = SparseLogreg(X_train, y_train, np.log(alpha), max_iter=max_iter, tol=tol)
diff_beta_logreg, diff_jac_logreg, n_iter_logreg, supp_id = linear_cv(X_train, y_train, np.log(alpha), model, clf.coef_, jac_star,
                                                                      max_iter=max_iter, tol=tol, compute_jac=True)

fig, axarr = plt.subplots(
    1, 2, sharex=False, sharey=False, figsize=[10, 4])
plt.figure()
axarr.flat[0].semilogy(range(n_iter_logreg+1), diff_beta_logreg, linewidth=2.0)
axarr.flat[1].semilogy(range(n_iter_logreg+1), diff_jac_logreg, linewidth=2.0)
axarr.flat[0].set_xlabel("epoch")
axarr.flat[0].set_ylabel("||beta - beta_star||")
axarr.flat[1].set_xlabel("epoch")
axarr.flat[1].set_ylabel("||jac - jac_Star||")
axarr.flat[0].axvline(x=supp_id[0], c='red', linestyle="--")
axarr.flat[1].axvline(x=supp_id[0], c='red', linestyle="--")
axarr.flat[0].set_title("Iterates convergence for the Logistic Regression")
axarr.flat[1].set_title("Jacobian convergence for the Logistic Regression")
plt.show(block=False)
