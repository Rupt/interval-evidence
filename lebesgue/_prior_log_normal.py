"""Implement the log-normal prior."""
import numba
import numpy

from . import _cephes_ndtr
from ._bayes import _Prior


def log_normal(mu: float, sigma: float) -> _Prior:
    """Return a log-normal prior for given mu, sigma.

    Arguments:
        mu: log-mean
        sigma: log-standard-deviation
    """
    mu = float(mu)

    sigma = float(sigma)
    if not sigma > 0:
        raise ValueError(sigma)

    tau = 1 / sigma
    return _Prior((mu, tau), _log_normal_between)


@numba.njit(cache=True)
def _log_normal_between(args, lo, hi):
    mu, tau = args
    if lo > 0:
        lo = numpy.log(lo)
    else:
        lo = -numpy.inf

    if hi > 0:
        hi = numpy.log(hi)
    else:
        hi = -numpy.inf

    lo = (lo - mu) * tau
    hi = (hi - mu) * tau

    return gaussian_dcdf(lo, hi)


@numba.njit(numba.float64(numba.float64, numba.float64), cache=True)
def gaussian_dcdf(lo, hi):
    """Return cdf(hi) - cdf(lo) with reduced truncation error."""
    offset = numpy.copysign(0.5, hi) - numpy.copysign(0.5, lo)

    flo = numpy.copysign(_cephes_ndtr.ndtr(-abs(lo)), lo)
    fhi = numpy.copysign(_cephes_ndtr.ndtr(-abs(hi)), hi)

    return flo - fhi + offset
