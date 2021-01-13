import cvxpy as cp
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer

torch.set_default_dtype(torch.double)
# np.set_printoptions(precision=3, suppress=True)


def enet_cvxpy(X, y, lambda_alpha, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    m, n = Xtrain.shape

    # set up variables and parameters
    beta = cp.Variable(n)
    lambda_cp = cp.Parameter(nonneg=True)
    alpha_cp = cp.Parameter(nonneg=True)

    # set up objective
    loss = (1/(2 * m)) * cp.sum(cp.square(Xtrain @ beta - ytrain))
    reg = lambda_cp * cp.norm1(beta) + alpha_cp * cp.sum_squares(beta) / 2
    objective = loss + reg

    # define problem
    problem = cp.Problem(cp.Minimize(objective))
    assert problem.is_dpp()

    # solve problem
    layer = CvxpyLayer(problem, [lambda_cp, alpha_cp], [beta])
    lambda_alpha_th = torch.tensor(lambda_alpha, requires_grad=True)
    beta_, = layer(lambda_alpha_th[0], lambda_alpha_th[1])

    # get test loss and it's gradient
    test_loss = (Xtest @ beta_ - ytest).pow(2).mean()
    test_loss.backward()

    val = test_loss.detach().numpy()
    grad = np.array(lambda_alpha_th.grad)
    return val, grad


def weighted_lasso_cvxpy(X, y, lambdas, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    m, n = Xtrain.shape

    # set up variables and parameters
    beta = cp.Variable(n)
    lambda_cp = cp.Parameter(shape=n, nonneg=True)

    # set up objective
    loss = (1/(2 * m)) * cp.sum(cp.square(Xtrain @ beta - ytrain))
    reg = lambda_cp @ cp.abs(beta)
    objective = loss + reg

    # define problem
    problem = cp.Problem(cp.Minimize(objective))
    assert problem.is_dpp()

    # solve problem
    layer = CvxpyLayer(problem, [lambda_cp], [beta])
    lambdas_th = torch.tensor(lambdas, requires_grad=True)
    beta_, = layer(lambdas_th)

    # get test loss and it's gradient
    test_loss = (Xtest @ beta_ - ytest).pow(2).mean()
    test_loss.backward()

    val = test_loss.detach().numpy()
    grad = np.array(lambdas_th.grad)
    return val, grad


def logreg_cvxpy(X, y, alpha, idx_train, idx_val):
    Xtrain, Xtest, ytrain, ytest = map(
        torch.from_numpy, [
            X[idx_train, :], X[idx_val], y[idx_train], y[idx_val]])

    m, n = Xtrain.shape

    # set up variables and parameters
    beta = cp.Variable(n)
    # X = cp.Parameter((m, n))
    # Y = cp.Parameter(m)
    alpha_cp = cp.Parameter(nonneg=True)

    # set up objective
    loss = cp.sum(cp.logistic(Xtrain @ beta) - cp.multiply(ytrain, Xtrain @ beta)) / m
    reg = alpha_cp * cp.norm(beta, 1)
    objective = loss + reg

    # define problem
    problem = cp.Problem(cp.Minimize(objective))
    assert problem.is_dpp()

    # solve problem
    # layer = CvxpyLayer(problem, parameters=[X, Y, alpha_cp], variables=[beta])
    layer = CvxpyLayer(problem, parameters=[alpha_cp], variables=[beta])
    alpha_th = torch.tensor(alpha, requires_grad=True)
    # beta_, = layer(Xtrain, ytrain, alpha_th)
    beta_, = layer(alpha_th)

    # get test loss and it's gradient
    loss_th = torch.nn.modules.loss.BCEWithLogitsLoss(reduction='mean')
    test_loss = loss_th(Xtest @ beta_, ytest)
    test_loss.backward()

    val = test_loss.detach().numpy()
    grad = np.array(alpha_th.grad)
    return val, grad
