import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
import pandas
from itertools import product
import blitzl1
# from sklearn.linear_model import Lasso as Lasso_sk
# from sklearn.linear_model import LogisticRegression
# from celer import LogisticRegression
from scipy.sparse.linalg import cg
from scipy.sparse import csc_matrix
from sparse_ho.models import LassoGradSearch, SparseLogregGradSearch
# , SparseLogreg
# import matplotlib.pyplot as plt

from celer import Lasso as Lasso_cel
# from sparse_ho.utils import sigma
from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.datasets.real import load_libsvm

p_alphas = {}
p_alphas["leu", "lasso"] = 0.01
p_alphas["rcv1_train", "lasso"] = 0.075
p_alphas["news20", "lasso"] = 0.3
p_alphas["finance", "lasso"] = 0.3
p_alphas["real-sim", "lasso"] = 0.1

p_alphas["leu", "logreg"] = 0.1
p_alphas["rcv1_train", "logreg"] = 0.25
p_alphas["news20", "logreg"] = 0.8
p_alphas["finance", "logreg"] = 0.3
p_alphas["real-sim", "logreg"] = 0.15

max_iters = {}
max_iters["leu"] = 2000
max_iters["rcv1_train"] = 200
max_iters["news20"] = 100
max_iters["real-sim"] = 100

# dataset_names = [""]
# dataset_names = ["news20"]
#
dataset_names = ["leu"]
# dataset_names = ["rcv1_train"]
# dataset_names = ["leu"]
# dataset_names = ["real-sim"]
# dataset_names = ["finance"]
# dataset_names = ["leu"]
model_names = ["logreg"]
# model_names = ["lasso"]


def linear_cv(
        dataset_name, tol=1e-3, compute_jac=True, model_name="lasso"):

    X, y = load_libsvm(dataset_name)
    X = csc_matrix(X)
    n_samples, n_features = X.shape
    p_alpha = p_alphas[dataset_name, model_name]

    max_iter = max_iters[dataset_name]
    if model_name == "lasso":
        model = Lasso(X, y, 0, max_iter=max_iter, tol=tol)
    elif model_name == "logreg":
        model = SparseLogreg(X, y, 0, max_iter=max_iter, tol=tol)

    alpha_max = np.exp(model.compute_alpha_max())

    alpha = p_alpha * alpha_max
    if model_name == "lasso":
        clf = Lasso_cel(
            alpha=alpha, fit_intercept=False, warm_start=True,
            tol=tol * norm(y) ** 2 / 2, max_iter=10000)
        clf.fit(X, y)
        beta_star = clf.coef_
        mask = beta_star != 0
        dense = beta_star[mask]
    elif model_name == "logreg":
        # clf = LogisticRegression(
        #     penalty='l1', C=(1 / (alpha * n_samples)),
        #     fit_intercept=False,
        #     warm_start=True, max_iter=10000,
        #     tol=tol, verbose=True).fit(X, y)
        # clf = LogisticRegression(
        #     penalty='l1', C=(1 / (alpha * n_samples)),
        #     fit_intercept=False,
        #     warm_start=True, max_iter=10000,
        #     tol=tol, verbose=True,
        #     solver='liblinear').fit(X, y)
        # beta_star = clf.coef_[0]
        # import ipdb; ipdb.set_trace()

        # import ipdb; ipdb.set_trace()
        blitzl1.set_use_intercept(False)
        blitzl1.set_tolerance(1e-32)
        blitzl1.set_verbose(True)
        # blitzl1.set_min_time(60)
        prob = blitzl1.LogRegProblem(X, y)
        # # lammax = prob.compute_lambda_max()
        clf = prob.solve(alpha * n_samples)
        beta_star = clf.x
        mask = beta_star != 0
        mask = np.array(mask)
        dense = beta_star[mask]
    # import ipdb; ipdb.set_trace()
    # if model == "lasso":
    v = - n_samples * alpha * np.sign(beta_star[mask])
    mat_to_inv = model.get_hessian(mask, dense, np.log(alpha))
    # mat_to_inv = X[:, mask].T  @ X[:, mask]

    jac_temp = cg(mat_to_inv, v, tol=1e-10)
    jac_star = np.zeros(n_features)
    jac_star[mask] = jac_temp[0]
    # elif model == "logreg":
    #     v = - n_samples * alpha * np.sign(beta_star[mask])

    log_alpha = np.log(alpha)

    list_beta, list_jac = get_beta_jac_iterdiff(
        X, y, log_alpha, model, save_iterates=True, tol=tol,
        max_iter=max_iter, compute_jac=compute_jac)

    diff_beta = norm(list_beta - beta_star, axis=1)
    diff_jac = norm(list_jac - jac_star, axis=1)

    supp_star = beta_star != 0
    n_iter = list_beta.shape[0]
    for i in np.arange(n_iter)[::-1]:
        supp = list_beta[i, :] != 0
        # import ipdb; ipdb.set_trace()
        if not np.all(supp == supp_star):
            supp_id = i + 1
            break
        supp_id = 0

    return dataset_name, p_alpha, diff_beta, diff_jac, n_iter, supp_id


# parameter of the algo
tol = 1e-16
# max_iter = 100
# max_iter = 10000

# dataset_name = "news20"
# dataset_name = "rcv1_train"
# dataset_name = "leu"
p_alpha = 0.01
# p_alpha = 0.3

# diff_beta, diff_jac, n_iter, supp_id = linear_cv(
# results = linear_cv(
#     dataset_name, p_alpha,
#     max_iter=max_iter, tol=tol, compute_jac=True)

print("enter sequential")
backend = 'loky'
n_jobs = 1
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(linear_cv)(
        dataset_name, tol=tol, compute_jac=True, model_name=model_name)
    for dataset_name, model_name in product(dataset_names, model_names))
print('OK finished parallel')

df = pandas.DataFrame(results)

df.columns = [
    'dataset_name', 'p_alpha', 'diff_beta', 'diff_jac', 'n_iter',
    'supp_id']

for dataset_name in dataset_names:
    for model_name in model_names:
        df[df['dataset_name'] == dataset_name].to_pickle(
            "%s_%s.pkl" % (dataset_name, model_name))


# Same for the Logistic regression

# n_samples = 100
# n_features = 1000
# SNR = 3.0
# seed = 10
# tol = 1e-14
# max_iter = 10000

# X_train, y_train = datasets.make_classification(
#     n_samples=n_samples,
#     n_features=n_features, n_informative=50,
#     random_state=110, flip_y=0.1, n_redundant=0)

# y_train[y_train == 0.0] = -1.0

# X_val, y_val = datasets.make_classification(
#     n_samples=n_samples,
#     n_features=n_features, n_informative=50,
#     random_state=122, flip_y=0.1, n_redundant=0)


# alpha_max = norm(X_train.T @ y_train, ord=np.inf) / (2 * n_samples)
# alpha = 0.05 * alpha_max

# clf = LogisticRegression(penalty="l1", tol=1e-12, C=(
#                          1 / (alpha * n_samples)), fit_intercept=False, max_iter=100000,
#                          solver="liblinear")
# clf.fit(X_train, y_train)

# beta_star_logreg = clf.coef_
# supp_sk = beta_star_logreg != 0
# supp_sk = supp_sk[0, :].T
# dense_sk = beta_star_logreg[0, supp_sk].T

# v = - n_samples * alpha * np.sign(dense_sk)
# r = y_train * (X_train[:, supp_sk] @ dense_sk)
# mat_to_inv = X_train[:, supp_sk].T  @ np.diag(sigma(r) * (1 - sigma(r))) @ X_train[:, supp_sk]

# jac_temp = cg(mat_to_inv, v, atol=1e-7)
# jac_star = np.zeros(n_features)
# jac_star[supp_sk] = jac_temp[0]

# model = SparseLogreg(X_train, y_train, np.log(alpha), max_iter=max_iter, tol=tol)
# diff_beta_logreg, diff_jac_logreg, n_iter_logreg, supp_id = linear_cv(X_train, y_train, np.log(alpha), model, clf.coef_, jac_star,
#                                                                       max_iter=max_iter, tol=tol, compute_jac=True)

# fig, axarr = plt.subplots(
#     1, 2, sharex=False, sharey=False, figsize=[10, 4])
# plt.figure()
# axarr.flat[0].semilogy(range(n_iter_logreg+1), diff_beta_logreg, linewidth=2.0)
# axarr.flat[1].semilogy(range(n_iter_logreg+1), diff_jac_logreg, linewidth=2.0)
# axarr.flat[0].set_xlabel("epoch")
# axarr.flat[0].set_ylabel("||beta - beta_star||")
# axarr.flat[1].set_xlabel("epoch")
# axarr.flat[1].set_ylabel("||jac - jac_Star||")
# axarr.flat[0].axvline(x=supp_id[0], c='red', linestyle="--")
# axarr.flat[1].axvline(x=supp_id[0], c='red', linestyle="--")
# axarr.flat[0].set_title("Iterates convergence for the Logistic Regression")
# axarr.flat[1].set_title("Jacobian convergence for the Logistic Regression")
# plt.show(block=False)
