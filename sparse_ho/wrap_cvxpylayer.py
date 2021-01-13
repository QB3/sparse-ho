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


# from sparse_ho.datasets.synthetic import get_synt_data

# n_samples = 10
# n_features = 10
# n_active = 5
# SNR = 3
# rho = 0.1

# X, y, beta_star, noise, sigma_star = get_synt_data(
#     dictionary_type="Toeplitz", n_samples=n_samples,
#     n_features=n_features, n_times=1, n_active=n_active, rho=rho,
#     SNR=SNR, seed=0)

# idx_train = np.arange(0, n_features//2)
# idx_val = np.arange(n_features//2, n_features)

# alpha_max = (np.abs(X[idx_train, :].T @ y[idx_train])).max() / n_samples
# p_alpha = 0.8
# alpha = p_alpha * alpha_max
# lambdas = alpha * np.ones(n_features)

# val, grad = wLasso_cvxpy(X, y, lambdas, idx_train, idx_val)
