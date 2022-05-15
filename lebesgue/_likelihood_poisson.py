"""Implement the Poisson likelihood function and friends."""
from ._bayes import Likelihood
from ._invg import ginterval


def poisson(n: int) -> Likelihood:
    """Return a Poisson likelihood for n observed events.

    Although n is an integer, we store it as a float to reduce casting later.

    Arguments:
        n: number of observed events; non-negative integer

    """
    if not n == float(int(n)):
        raise ValueError(n)
    return gamma1(int(n))


def gamma1(shape: float) -> Likelihood:
    """Return a Gamma likelihood for x=1 observed and given shape.

    Possibly useful as a continuous extrapolation of Poisson.

    Arguments:
        shape: `alpha' argument

    """
    shape = float(shape)
    if not shape >= 0:
        raise ValueError(shape)
    return Likelihood(shape, ginterval)
