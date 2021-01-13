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
    n = X.shape[1]
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])
    Xtrain.requires_grad_(True)

    m = Xtrain.shape[0]

    beta = cp.Variable(n)

    lambd = cp.Parameter(nonneg=True)

    log_likelihood = cp.sum(
        cp.logistic(X @ beta) - cp.multiply(y, X @ beta)
    )
    loss = log_likelihood / m
    reg = lambd * cp.norm(beta, 1)
    problem = cp.Problem(cp.Minimize(loss + reg))
    # problem = cp.Problem(objective)
    assert problem.is_dpp()
    cvxpylayer = CvxpyLayer(problem, parameters=[lambd], variables=[beta])
    lambd = torch.tensor(alpha, requires_grad=True)
    # solve the problem
    solution, = cvxpylayer(lambd)
    loss = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean')
    val = loss(Xtest @ solution, ytest)
    val.backward()
    grad = lambd.grad
    val = val.detach().numpy()
    return val, grad
