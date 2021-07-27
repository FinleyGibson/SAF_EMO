import numpy as np
from testsuite.utilities import Pareto_split
from scipy.stats import norm


def optional_inversion(f):
    """decorator to invert the value of a function, and turn maximisation
    problem to minimization problem. Invoke by passing a keyword argument
    invert=True to the decorated function"""
    def wrapper(*args, **kwargs):
        try:
            if kwargs["invert"] is True:
                del(kwargs["invert"])
                return -f(*args, **kwargs)
            else:
                del(kwargs["invert"])
                return f(*args, **kwargs)
        except KeyError:
            return f(*args, **kwargs)
    return wrapper


def scalar_expected_improvement(mu, var, y, xi=0.01, invert=False):
    """
    computes the expected improvement of the putative solution of a
    prediction based on the mean (mu) and variance (var) of the
    prediciton over the previous observastions in x_observed according
    to the provided surrogate model.

    :param float mu: prediction mean
    :param float var:
    :param np.array y:
    :param float xi:
    :return float: expected improvement
    """
    if invert:
        muq = -1*mu
        yq = -1*y
    else:
        muq = mu
        yq = y

    mu_opt = np.max(yq)

    sigma = var**0.5

    with np.errstate(divide='warn'):
        imp = muq - mu_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def UCB(mu, var, beta=0.05, invert=False):
    ucb =  mu + beta * var**0.5
    if invert:
        return -ucb
    else:
        return ucb


def PI(mu, var, y, xi=0.01, invert=False):

    if invert:
        muq = -1*mu
        yq = -1*y
    else:
        muq = mu
        yq = y

    mu_opt = np.max(yq)

    sigma = var**0.5

    imp = muq - mu_opt - xi
    Z = imp / sigma
    pi = norm.cdf(Z) + sigma * norm.pdf(Z)
    return norm.cdf(Z)


def _mean_signed_improvement(x_put, x_observed, surrogate):

    mu, var = surrogate.predict(x_observed)
    mu_put, var_put = surrogate.predict(x_put)

    mu_opt = np.max(mu)
    return mu_put-mu_opt


@optional_inversion
def saf(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Calculates summary attainment front distances.
    Calculates the distance of n, m-dimensional points X from the
    summary attainment front defined by points P

    :param np.array p: points in the pareto front, shape[?,m]
    :param np.array y: points for which the distance to the summary
    attainment front is to be calculated, shape[n,m]

    :return np.array: numpy array of saf distances between points in
    X and saf defined by P, shape[X.shape]
    """
    # check dimensionality of P is the same as that for X
    assert p.shape[1] == y.shape[1]

    D = np.zeros((y.shape[0], p.shape[0]))

    for i, p in enumerate(p):
        D[:, i] = np.max(p - y, axis=1).reshape(-1)
    Dq = np.min(D, axis=1)
    return Dq


def saf_mu(x_put, surrogate, z_pop: np.array):
    """
    :param x_put: putitive solution to be tested
    :param surrogate: surrogate
    :param z_pop: population of objective space observations so far

    :return: returns the saf estimate of solution efficiacy based on the
    surrogate predictions and the current
    observations
    """
    # split dominated and non-dominated solutions so far
    p, d = Pareto_split(z_pop)

    # get mean predictions
    mu = surrogate.predict(x_put)[0]
    # get saf of mean predictions
    alpha = saf(mu, p)  # invert to minimize problem
    return alpha


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from testsuite.surrogates import GP

    def test_function(x):
        return np.sin(x)

    np.random.seed(1)
    x = np.linspace(-5, 5, 200).reshape(-1, 1)
    y = test_function(x)
    xtr = np.random.uniform(-5, 5, 4).reshape(-1, 1)
    ytr = test_function(xtr)

    model = GP(scaled=True)
    model.update(xtr, ytr)
    model_y, model_var = model.predict(x)


    fig0, [ax00, ax01] = plt.subplots(2, 1)
    ax00.plot(x, y, c="grey", label="objective function")
    ax00.plot(x, model_y, c="C0", alpha=1., label="surrogate model")
    ax00.fill_between(x.flatten(),
                     (model_y-(2*np.sqrt(model_var))).flatten(),
                     (model_y+(2*np.sqrt(model_var))).flatten(),
                     color="C0", alpha=0.3)
    ax00.scatter(xtr, ytr, marker="x", c="C0", label="observations")
    ax00.legend()
    ei = scalar_expected_improvement(model_y, model_var, model.get_y(), invert=False) # use get_y() as y is scaled
    pi = PI(model_y, model_var, model.get_y(), invert=False)# use get_y() as y is scaled
    beta = 0.5
    ucb = UCB(model_y, model_var, beta=beta, invert=False)
    ax01.plot(x, ei, c="C4", alpha=0.7, label="ei")
    ax01.plot(x, ucb, c="C5", alpha=0.7, label="ucb")
    ax01.plot(x, pi, c="C6", alpha=0.7, label="pi")

    ax01.legend()


    fig1, [ax10, ax11] = plt.subplots(2, 1)
    ax10.plot(x, y, c="grey", label="objective function")
    ax10.plot(x, model_y, c="C0", alpha=1., label="surrogate model")
    ax10.fill_between(x.flatten(),
                      (model_y-(2*np.sqrt(model_var))).flatten(),
                      (model_y+(2*np.sqrt(model_var))).flatten(),
                      color="C0", alpha=0.3)
    ax10.scatter(xtr, ytr, marker="x", c="C0", label="observations")
    ax10.legend()
    ei_inv = scalar_expected_improvement(model_y, model_var, model.get_y(), invert=True) # use get_y() as y is scaled
    pi_inv = PI(model_y, model_var, model.get_y(), invert=True)# use get_y() as y is scaled
    ucb_inv = UCB(model_y, model_var, beta=beta, invert=True)
    ax11.plot(x, ei_inv, c="C4", alpha=0.7, label="ei")
    ax11.plot(x, ucb_inv, c="C5", alpha=0.7, label="ucb")
    ax11.plot(x, pi_inv, c="C6", alpha=0.7, label="pi")
    ax11.legend()
    plt.show()
