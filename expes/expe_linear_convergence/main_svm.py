import numpy as np
from numpy.linalg import norm
from joblib import Parallel, delayed
import pandas
from scipy.sparse.linalg import cg
from sparse_ho.models import SVM
from cvxopt import spmatrix, matrix
from cvxopt import solvers
from sparse_ho.forward import get_beta_jac_iterdiff
from sparse_ho.datasets.real import load_libsvm


dataset_names = ["leu"]
# dataset_names = ["finance"]
# dataset_names = ["leu", "rcv1_train", "news20"]
Cs = {}
Cs["leu"] = 0.01
Cs["rcv1_train"] = 0.01
Cs["news20"] = 0.01
Cs["finance"] = 0.01
Cs["real-sim"] = 0.01


def scipy_sparse_to_spmatrix(A):
    coo = A.tocoo()
    SP = spmatrix(coo.data, coo.row.tolist(), coo.col.tolist())
    return SP


def linear_cv(dataset_name, max_iter=1000, tol=1e-3, compute_jac=True):
    X, y = load_libsvm(dataset_name)
    X = X.tocsr()
    n_samples, n_features = X.shape
    C = Cs[dataset_name]
    # Computation of dual solution of SVM via cvxopt
    Q = (X.multiply(y[:, np.newaxis]))  @  (X.multiply(y[:, np.newaxis])).T
    # Quadratic term
    Q_cvx = scipy_sparse_to_spmatrix(Q)
    # Linear term
    L = np.repeat(-1.0, Q.shape[0])
    L = matrix(L)

    # matrix inequality constraints
    G = np.zeros((2 * n_samples, n_samples))
    G[0:n_samples, :] = -np.eye(n_samples)
    G[n_samples:(2 * n_samples), :] = np.eye(n_samples)
    G = matrix(G)

    # vector inequality constraints
    h = np.zeros(2 * n_samples)
    h[n_samples:(2 * n_samples)] = C
    h = matrix(h)

    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-16
    solvers.options['feastol'] = 1e-16
    solvers.options['reltol'] = 1e-16
    sol = solvers.qp(Q_cvx, L, G, h, None, None)
    beta_star = sol['x'][:n_samples]
    beta_star = np.array(beta_star).flatten()

    full_supp = np.logical_and(np.logical_not(np.isclose(beta_star, 0)), np.logical_not(np.isclose(beta_star, C)))

    v = np.array((np.eye(n_samples, n_samples) - Q)[np.ix_(full_supp, np.isclose(beta_star, C))] @ (np.ones((np.isclose(beta_star, C)).sum()) * C))
    v = np.squeeze(v)
    jac_dense = cg(Q[np.ix_(full_supp, full_supp)], v, tol=1e-14)
    jac_star = np.zeros(n_samples)
    jac_star[full_supp] = jac_dense[0]
    jac_star[np.isclose(beta_star, C)] = C

    model = SVM(X, y, np.log(C), max_iter=max_iter, tol=tol)
    list_beta, list_jac = get_beta_jac_iterdiff(
        X, y, np.log(C), model, save_iterates=True, tol=tol,
        max_iter=max_iter, compute_jac=compute_jac)
    diff_beta = norm(list_beta - beta_star, axis=1)
    diff_jac = norm(list_jac - jac_star, axis=1)

    full_supp_star = full_supp
    n_iter = list_beta.shape[0]
    for i in np.arange(n_iter)[::-1]:
        full_supp = np.logical_and(np.logical_not(np.isclose(list_beta[i, :], 0)), np.logical_not(np.isclose(list_beta[i, :], C)))
        # import ipdb; ipdb.set_trace()
        if not np.all(full_supp == full_supp_star):
            supp_id = i + 1
            break
        supp_id = 0

    return dataset_name, C, diff_beta, diff_jac, n_iter, supp_id


# parameter of the algo
tol = 1e-16
max_iter = 10000

print("enter sequential")
backend = 'loky'
n_jobs = 1
results = Parallel(n_jobs=n_jobs, verbose=100, backend=backend)(
    delayed(linear_cv)(
        dataset_name,
        max_iter=max_iter, tol=tol, compute_jac=True)
    for dataset_name in dataset_names)
print('OK finished parallel')

df = pandas.DataFrame(results)

df.columns = [
    'dataset_name', 'p_alpha', 'diff_beta', 'diff_jac', 'n_iter',
    'supp_id']

for dataset_name in dataset_names:
    df[df['dataset_name'] == dataset_name].to_pickle(
        "%s_svm.pkl" % dataset_name)
