import numpy as np
from typing import Union
from testsuite.utilities import Pareto_split
from testsuite.surrogates import MultiSurrogate, MonoSurrogate
from testsuite.scalarisers import saf
from scipy.stats import norm


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
        mu_opt = np.min(y)
    else:
        mu_opt = np.max(y)

    sigma = var**0.5

    with np.errstate(divide='warn'):
        if invert:
            imp = -mu + mu_opt + xi
        else:
            imp = mu - mu_opt - xi
        Z = imp / sigma
        ei = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        ei[sigma == 0.0] = 0.0
    return ei


def _mean_signed_improvement(x_put, x_observed, surrogate):
    mu, var = surrogate.predict(x_observed)
    mu_put, var_put = surrogate.predict(x_put)

    mu_opt = np.max(mu)
    return mu_put-mu_opt


def saf_mu(x_put: np.ndarray, surrogate: Union[MonoSurrogate, MultiSurrogate],
           z_pop: np.array) -> np.ndarray:
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


def saf_ei(x_put: np.ndarray, surrogate: Union[MonoSurrogate, MultiSurrogate],
           z_pop: np.array, n_samples: int=10000, return_samples=False) \
        -> Union[np.ndarray, tuple]:
    # TODO implement saf_calc for multiple points simaltaneously
    return _single_saf_ei(x_put, surrogate, z_pop)


def _single_saf_ei(x_put: np.ndarray, surrogate: Union[MonoSurrogate, MultiSurrogate],
           z_pop: np.array, n_samples: int=10000, return_samples=False) \
        -> Union[np.ndarray, tuple]:

    if x_put.ndim==1:
        x_put = x_put.reshape(1,-1)

    assert(x_put.ndim==2)
    assert x_put.shape[0]==1
    # TODO implement multiple point saf_ei computation simultaneously.

    # split dominated and non-dominated solutions so far
    p, d = Pareto_split(z_pop)

    mu_put, var_put = surrogate.predict(x_put)
    mu_put = mu_put.reshape(-1)
    var_put = var_put.reshape(-1)

    # best saf value observed so far.
    f_star = np.max(saf(z_pop, p))  # should be 0 in current cases

    # sample from surrogate
    samples = np.array([np.random.normal(mu, var**0.5, n_samples)
                        for mu, var in zip(mu_put, var_put)]).T
    saf_samples = saf(samples, p)
    saf_samples[saf_samples > f_star] = f_star

    if return_samples:
        return samples, saf_samples
    else:
        return np.mean(saf_samples)


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from testsuite.surrogates import GP

    def test_function(x):
        return np.sin(x)

    x = np.linspace(-5, 5, 200).reshape(-1, 1)
    y = test_function(x)
    xtr = np.random.uniform(-5, 5, 7).reshape(-1, 1)
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

    # use get_y() as y is scaled
    ei = scalar_expected_improvement(model_y, model_var, model.get_y())
    ax01.plot(x, ei, c="C4", alpha=0.7, label="ei")
    ei_inv = scalar_expected_improvement(model_y, model_var, model.get_y(),
                                         invert=True)
    ax01.plot(x, ei_inv, c="C2", alpha=0.7, label="ei inverted")
    ax01.legend()

    plt.show()
