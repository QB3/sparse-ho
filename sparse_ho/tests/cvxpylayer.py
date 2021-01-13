import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.double)
# np.set_printoptions(precision=3, suppress=True)


def enet_cvxpy(X, y, lambda_alpha, idx_train, idx_val):
    N, n = X.shape
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])
    Xtrain.requires_grad_(True)

    m = Xtrain.shape[0]

    # set up variables and parameters
    a = cp.Variable(n)
    # b = cp.Variable()
    X = cp.Parameter((m, n))
    Y = cp.Parameter(m)
    lam = cp.Parameter(nonneg=True)
    alpha = cp.Parameter(nonneg=True)

    # set up objective
    loss = (1/(2 * m))*cp.sum(cp.square(X @ a - Y))
    reg = lam * cp.norm1(a) + alpha * cp.sum_squares(a) / 2
    objective = loss + reg

    # set up constraints
    constraints = []

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # convert into pytorch layer in one line
    fit_lr = CvxpyLayer(prob, [X, Y, lam, alpha], [a])
    # this object is now callable with pytorch tensors
    fit_lr(Xtrain, ytrain, torch.zeros(1), torch.zeros(1))

    # sweep over values of alpha, holding lambda=0, evaluating the gradient
    # along the way

    lambda_alpha_tch = torch.tensor(
        lambda_alpha, requires_grad=True)
    lambda_alpha_tch.grad = None
    a_tch = fit_lr(
        Xtrain, ytrain, lambda_alpha_tch[0], lambda_alpha_tch[1])

    test_loss = (Xtest @ a_tch[0] - ytest).pow(2).mean()
    test_loss.backward()
    # test_losses.append(test_loss.item())
    grad_alpha_lambda = lambda_alpha_tch.grad
    val = test_loss.detach().numpy()
    grad = np.array(grad_alpha_lambda)
    return val, grad


def wLasso_cvxpy(X, y, lambdas, idx_train, idx_val):
    N, n = X.shape
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])
    Xtrain.requires_grad_(True)

    m = Xtrain.shape[0]

    # set up variables and parameters
    a = cp.Variable(n)
    # b = cp.Variable()
    X = cp.Parameter((m, n))
    Y = cp.Parameter(m)
    lam = cp.Parameter(shape=n, nonneg=True)

    # set up objective
    loss = (1/(2 * m))*cp.sum(cp.square(X @ a - Y))
    reg = lam @ cp.abs(a)
    # reg = cp.norm1(cp.multiply(lam, a))
    objective = loss + reg

    # set up constraints
    constraints = []

    prob = cp.Problem(cp.Minimize(objective), constraints)

    # convert into pytorch layer in one line
    fit_lr = CvxpyLayer(prob, [X, Y, lam], [a])
    # this object is now callable with pytorch tensors
    fit_lr(Xtrain, ytrain, torch.zeros(n))

    # sweep over values of alpha, holding lambda=0, evaluating the gradient
    # along the way

    lambda_alpha_tch = torch.tensor(lambdas, requires_grad=True)
    lambda_alpha_tch.grad = None
    a_tch = fit_lr(
        Xtrain, ytrain, lambda_alpha_tch)

    test_loss = (Xtest @ a_tch[0] - ytest).pow(2).mean()
    test_loss.backward()
    # test_losses.append(test_loss.item())
    grad_alpha_lambda = lambda_alpha_tch.grad
    val = test_loss.detach().numpy()
    grad = np.array(grad_alpha_lambda)
    return val, grad


def logreg_cvxpy(X, y, alpha, idx_train, idx_val):
    N, n = X.shape
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])
    Xtrain.requires_grad_(True)

    m = Xtrain.shape[0]
    n_samples_test = len(ytest)
    # set up variables and parameters
    # b = cp.Variable()
    beta = cp.Variable(n)
    # X = cp.Parameter((m, n))
    # y = cp.Parameter(m)
    # lambd = cp.Parameter(nonneg=True)
    # log_likelihood = cp.sum(cp.logistic(X @ beta) - cp.multiply(y, X @ beta))
    # # log_likelihood -=
    # problem = cp.Problem(
    #     cp.Minimize(log_likelihood / n + lambd * cp.norm1(beta)))
    # # problem = cp.Problem(objective)
    # assert problem.is_dpp()
    # fit_lr = CvxpyLayer(problem, parameters=[lambd], variables=[beta])
    # # fit_lr = CvxpyLayer(problem, parameters=[lam], variables=[a])
    # # convert into pytorch layer in one line
    # # fit_lr = CvxpyLayer(prob, [X, y, lam], [a])
    # # this object is now callable with pytorch tensors
    # # fit_lr(Xtrain, ytrain, torch.tensor(alpha))

    # # sweep over values of alpha, holding lambda=0, evaluating the gradient
    # # along the way

    # alpha_tch = torch.tensor(alpha, requires_grad=True)
    # alpha_tch.grad = None
    # a_tch = fit_lr(alpha_tch)

    # import ipdb; ipdb.set_trace()
    # test_loss = cp.sum(
    #     cp.logistic(Xtest @ a_tch[0]) - cp.multiply(ytest, Xtest @ a_tch[0]))
    # test_loss.backward()
    # # test_losses.append(test_loss.item())
    # grad_alpha_lambda = lambd.grad
    # val = test_loss.detach().numpy()
    # grad = np.array(grad_alpha_lambda)
    lambd = cp.Parameter(nonneg=True)
    # objective = cp.Minimize(
    #     0.5 * cp.sum(cp.log(1 + cp.exp(-y * (X @ beta)))) + lambd * cp.norm1(beta)
    #     # 0.5 * cp.norm2(X @ beta - y)**2 + lambd * cp.norm1(beta)
    # )
    log_likelihood = cp.sum(
        cp.logistic(X @ beta) - cp.multiply(y, X @ beta)
    )
    problem = cp.Problem(cp.Minimize(log_likelihood / n + lambd * cp.norm(
        beta, 1)))
    # problem = cp.Problem(objective)
    assert problem.is_dpp()
    cvxpylayer = CvxpyLayer(problem, parameters=[lambd], variables=[beta])
    lambd = torch.tensor(alpha, requires_grad=True)
    # solve the problem
    solution, = cvxpylayer(lambd)
    val = (cp.logistic(Xtest @ solution) - cp.multiply(
            ytest, Xtest @ solution)).mean()
    val.backward()
    grad = lambd.grad
    print(lambd.grad)
    # supp, dense, jac = get_beta_jac_fast_iterdiff(
    #     X, y, float(np.log(lambd.detach().numpy())),
    #     tol=1e-3, model=Lasso(estimator=None), tol_jac=1e-3)


    return val, grad


from sparse_ho.datasets.synthetic import get_synt_data


# Generate data
n_samples = 10
n_features = 10
X, y, _, _, sigma_star = get_synt_data(
    dictionary_type="Toeplitz", n_samples=n_samples,
    n_features=n_features, n_times=1, n_active=5, rho=0.1,
    SNR=3, seed=0)
idx_train = np.arange(0, n_features//2)
idx_val = np.arange(n_features//2, n_features)

# Set alpha for the Lasso
alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
p_alpha = 0.8
alpha = p_alpha * alpha_max
log_alpha = np.log(alpha)
tol = 1e-16

y = np.sign(y)
y[y == -1] = 0
logreg_cvxpy(X, y, alpha, idx_train, idx_val)

# import numpy as np
# import cvxpy as cp
# import torch
# from cvxpylayers.torch import CvxpyLayer
# from sparse_ho.algo.implicit_forward import get_beta_jac_fast_iterdiff
# from sparse_ho.models import Lasso
# n, m = 2, 3
# beta = cp.Variable(n)
# coef = np.random.randn(n)
# X = np.random.randn(m, n)
# y = X @ coef + 0.1 * np.random.randn(m)
# lambd = cp.Parameter(nonneg=True)
# # objective = cp.Minimize(
# #     0.5 * cp.sum(cp.log(1 + cp.exp(-y * (X @ beta)))) + lambd * cp.norm1(beta)
# #     # 0.5 * cp.norm2(X @ beta - y)**2 + lambd * cp.norm1(beta)
# # )
# y = np.sign(y)
# y[y == -1] = 0
# log_likelihood = cp.sum(
#     cp.logistic(X @ beta) - cp.multiply(y, X @ beta)
# )
# problem = cp.Problem(cp.Minimize(log_likelihood / n + lambd * cp.norm(beta, 1)))
# # problem = cp.Problem(objective)
# assert problem.is_dpp()
# cvxpylayer = CvxpyLayer(problem, parameters=[lambd], variables=[beta])
# lambd = torch.ones(1, requires_grad=True)
# # solve the problem
# solution, = cvxpylayer(lambd)
# solution.sum().backward()
# print(lambd.grad)
# supp, dense, jac = get_beta_jac_fast_iterdiff(
#     X, y, float(np.log(lambd.detach().numpy())),
#     tol=1e-3, model=Lasso(estimator=None), tol_jac=1e-3)
