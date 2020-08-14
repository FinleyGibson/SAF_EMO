import numpy as np
from testsuite.utilities import saf, Pareto_split


def saf_mu(x_put, surrogate, z_pop, *args, **kwargs):
    """
    :param x_put: putitive solution to be tested
    :param surrogates: list of surrogates
    :param z_pop: population of objective space observations so far
    :param args: collects any additional args supplied in erro
    :param kwargs: collects any additional args supplied in erro

    :return: returns the saf estimate of solution efficiacy based on the surrogate predictions and the current
    observations
    """
    # split dominated and non-dominated solutions so far
    p, d = Pareto_split(z_pop)

    # get mean predicitons
    mu = surrogate.predict(x_put)[0]
    # get saf of mean predictions
    alpha = saf(p, mu)
    return alpha
