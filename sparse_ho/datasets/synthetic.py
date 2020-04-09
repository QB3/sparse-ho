import numpy as np

from numpy.linalg import norm
from scipy.linalg import toeplitz
from sklearn.utils import check_random_state


def get_synt_data(
        dictionary_type="Gaussian", noise_type="Gaussian_iid", n_samples=20,
        n_features=20, n_times=30, n_active=3, rho=0.3,
        SNR=1/7, seed=0):
    """Simulate artificial data.

    Parameters:
    ----------
    dictionary_type: string
        "Gaussian", "Toeplitz", real_me
    n_samples: int
        number of channels
    n_features: int
        number of potential sources.
    n_times: int
        number of time points
    n_active: int
        number of active sources
    rho: float
        coefficient of correlation for the Toeplitz-corralted dictionary
    SNR: float
        Signal to Noise Ratio
    seed: int

    Returns
    -------
    X: np.array, shape (n_samples, n_features)
        dictionary/gain matrix
    Y: np.array, shape (n_samples, n_times)
        data observed
    B_star: np.array, shape (n_features, n_times))
        real regression coefficients
    """
    rng = check_random_state(seed)

    X = get_dictionary(
        dictionary_type, n_samples=n_samples,
        n_features=n_features, rho=rho, seed=seed)

    rng = check_random_state(seed)
    # creates the signal XB
    B_star = np.zeros([n_features, n_times])
    supp = rng.choice(n_features, n_active, replace=False)
    B_star[supp, :] = 1
    # B_star[supp, :] = rng.randn(n_active, n_times)
    assert (norm(B_star, axis=1) != 0).sum() == n_active

    Y = X @ B_star
    noise = rng.randn(n_samples, n_times)
    sigma_star = norm(Y, ord='fro') / (norm(noise, ord='fro') * SNR)
    noise *= sigma_star
    Y += noise
    # B_dns = B_star[supp, :]
    if n_times == 1:
        return X, Y[:, 0], B_star[:, 0], noise, sigma_star
    else:
        return X, Y, B_star, noise, sigma_star


def get_dictionary(
        dictionary_type, n_samples=20, n_features=30, rho=0.3, seed=0):
    rng = check_random_state(seed)
    if dictionary_type == "real_dico":
        raise NotImplementedError("No dictionary '{}' in nolhyps"
                                  .format(dictionary_type))
    elif dictionary_type == 'Toeplitz':
        X = get_toeplitz_dictionary(
            n_samples=n_samples, n_features=n_features, rho=rho, seed=seed)
    elif dictionary_type == 'Gaussian':
        X = rng.randn(n_samples, n_features)
    else:
        raise NotImplementedError("No dictionary '{}' in nolhyps"
                                  .format(dictionary_type))
    normalize(X)
    return X


def normalize(X):
    """Normalize each each feature of X in place.
    """
    for i in range(X.shape[1]):
        X[:, i] /= norm(X[:, i])
    return X


def get_toeplitz_dictionary(
        n_samples=20, n_features=30, rho=0.3, seed=0):
    """This function returns a toeplitz dictionnary phi.

    Maths formula:
    S = toepltiz(\rho ** [|0, n_features-1|], \rho ** [|0, n_features-1|])
    X[:, i] sim mathcal{N}(0, S).

    Parameters
    ----------
    n_samples: int
        number of channels/measurments in your problem
    n_labels: int
        number of labels/atoms in your problem
    rho: float
        correlation matrix

    Results
    -------
    X : array, shape (n_samples, n_labels)
        The dictionary.
    """
    rng = check_random_state(seed)
    vect = rho ** np.arange(n_features)
    covar = toeplitz(vect, vect)
    X = rng.multivariate_normal(np.zeros(n_features), covar, n_samples)
    return X


def get_synthetic_data(
        rng, n_samples=120, n_features=300, n_times=100, n_active=4,
        dictionary_type="Gaussian", rho=0.3, scale=False):
    """This is an old function of Yousra, 'stc_mind' object is needed in the
    folder where the script is launched.
    """
    X = get_dictionary(
        dictionary_type, n_samples=n_samples, n_features=n_features, rho=rho)
    X /= X.std(axis=0)
    B = np.zeros((n_features, n_times))
    supp = rng.choice(n_features, n_active, replace=False)
    for i, s in enumerate(supp):
        B[s, :] = np.sin(np.linspace(0, 2 * np.pi, n_times))
        B[s, :] *= np.random.randint(1, 10)
        # B[s, :] = stc.data[i]

    Y = np.dot(X, B)
    E = 0.1 * np.std(Y) * rng.randn(*Y.shape)
    Y += E
    if scale:
        rescale(X, Y)
    return X, Y, B


def rescale(X, Y):
    # n_samples, n_features, n_times = *X.shape, Y.shape[1]
    alpha_max = norm(X.T @ Y, axis=1).max()  # compute alpha_max
    alpha_max *= 0.01
    X /= alpha_max
