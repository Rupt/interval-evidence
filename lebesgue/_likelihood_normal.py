""" Implement the normal (Gaussian) likelihood. """
import numba
import numpy

from ._bayes import Likelihood


def normal(mu: float, sigma: float) -> Likelihood:
    """Return a Likelihood function with given mu and sigma."""
    mu = float(mu)
    sigma = float(sigma)
    if not sigma > 0:
        raise ValueError(sigma)

    return Likelihood((mu, sigma), _normal_interval)


@numba.njit(cache=True)
def _normal_interval(args, ratio):
    if ratio == 0:
        return -numpy.inf, numpy.inf
    chi = numpy.sqrt(-2 * numpy.log(ratio))
    mu, sigma = args
    return mu - sigma * chi, mu + sigma * chi
