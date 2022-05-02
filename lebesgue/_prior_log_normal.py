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

    return Prior(_LogNormal(mu, tau=1 / sigma))


@_core.jitclass
class _LogNormal:
    _mu: float
    _tau: float

    def __init__(self, mu, tau):
        self._mu = mu
        self._tau = tau

    def _between(self, lo, hi):
        if lo > 0:
            lo = numpy.log(lo)
        else:
            lo = -numpy.inf

        if hi > 0:
            hi = numpy.log(hi)
        else:
            hi = -numpy.inf

        lo = (lo - self._mu) * self._tau
        hi = (hi - self._mu) * self._tau

        return gaussian_dcdf(lo, hi)


@_core.jit(f8(f8, f8))
def gaussian_dcdf(lo, hi):
    """Return cdf(hi) - cdf(lo) with reduced truncation error."""
    offset = numpy.copysign(0.5, hi) - numpy.copysign(0.5, lo)

    flo = numpy.copysign(ndtr(-abs(lo)), lo)
    fhi = numpy.copysign(ndtr(-abs(hi)), hi)

    return flo - fhi + offset
