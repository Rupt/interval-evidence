""" Implement the Poisson likelihood function. """
from ._bayes import Likelihood
from ._invg import ginterval


def poisson(n: int) -> Likelihood:
    """Return a Poisson likelihood for n observed events.

    Although n is an integer, we store it as a float to reduce casting later.

    Arguments:
        n: number of observed events; non-negative integer

    """
    if not n == int(n) or not n == float(n):
        raise ValueError(n)
    n = float(int(n))
    if not n >= 0:
        raise ValueError(n)
    return Likelihood(n, ginterval)
