import numpy as np
from typing import Union
from testsuite.utilities import Pareto_split
from testsuite.surrogates import MultiSurrogate, MonoSurrogate
from testsuite.scalarisers import saf
from scipy.stats import norm


def _expected_improvement(x_put, x_observed, surrogate, xi=0.01):
    """
    computes the expected improvement of the putative solution x_put
    over the previous observastions in x_observed according to the
    provided surrogate model.

    :param np.ndarray x_put: putative solution (1, x_dims)
    :param np.ndarray x_observed: Prior observations
    (n_observations, x_dims)
    :param surrogate: surrogate model
    :param float xi:
    :return float: scalar value for expected improvement
    """
    # get surrogate model predictions for prior observations and putative
    # solution
    mu, var = surrogate.predict(x_observed)
    mu_put, var_put = surrogate.predict(x_put)

    mu_opt = np.max(mu)
    sigma = var_put**0.5

    with np.errstate(divide='warn'):
        imp = mu_put - mu_opt - xi
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

    ei = _expected_improvement(x, xtr, model)
    ax01.plot(x, ei, c="C4", alpha=0.7, label="expected improvement")
    ax01.legend()

    plt.show()
