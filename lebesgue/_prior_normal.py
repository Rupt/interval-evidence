"""Implement the normal (Gaussian) prior."""
import numba
import numpy

from . import _cephes_ndtr
from ._bayes import Prior
from ._prior_log import log


def normal(mu: float, sigma: float) -> Prior:
    """Return a normal prior for given mu, sigma.

    Arguments:
        mu: mean
        sigma: standard-deviation
    """
    mu = float(mu)
    sigma = float(sigma)
    if not sigma > 0:
        raise ValueError(sigma)

    tau = 1 / sigma
    return Prior((mu, tau), _normal_between)


def log_normal(mu: float, sigma: float) -> Prior:
    """Return a log-normal prior for given mu, sigma.

    Arguments:
        mu: log-mean
        sigma: log-standard-deviation
    """
    return log(normal(mu, sigma))


@numba.njit
def _normal_between(args, lo, hi):
    mu, tau = args
    return _gaussian_dcdf((lo - mu) * tau, (hi - mu) * tau)


@numba.njit
def _gaussian_dcdf(lo, hi):
    big_lo, smol_lo = _cephes_ndtr.ndtr_split(lo)
    big_hi, smol_hi = _cephes_ndtr.ndtr_split(hi)
    return big_hi - big_lo + (smol_hi - smol_lo)
