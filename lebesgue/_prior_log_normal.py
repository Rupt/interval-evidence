"""Implement the log-normal prior."""
import numpy
from numba import f8

from . import _core
from ._bayes import Prior
from ._cephes_ndtr import ndtr


def log_normal(mu, sigma):
    """Return a log-normal prior for given mu, sigma."""
    mu = float(mu)

    sigma = float(sigma)
    if not sigma > 0:
        raise ValueError(sigma)

    return Prior((mu, 1 / sigma), _log_normal_between)


@_core.jit(cache=True)
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


@_core.jit(f8(f8, f8), cache=True)
def gaussian_dcdf(lo, hi):
    """Return cdf(hi) - cdf(lo) with reduced truncation error."""
    offset = numpy.copysign(0.5, hi) - numpy.copysign(0.5, lo)

    flo = numpy.copysign(ndtr(-abs(lo)), lo)
    fhi = numpy.copysign(ndtr(-abs(hi)), hi)

    return flo - fhi + offset
