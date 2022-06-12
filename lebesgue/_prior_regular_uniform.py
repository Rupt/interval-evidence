"""A regularly piece-wise uniform prior."""
from collections.abc import Sequence

import numba
import numpy

from ._bayes import Prior


def regular_uniform(start: float, stop: float, log_rates: Sequence) -> Prior:
    """Return a Prior with piecewise uniform denisty from start to stop.

    Arguments:
        start: low edge
        stop: high edge
        log_rates: log density [+ c] in each bin

            Bin edges are at numpy.linspace(start, stop, len(log_rates) + 1)
    """
    stop = float(stop)
    start = float(start)
    if not start < stop:
        raise ValueError((start, stop))

    log_rates = numpy.array(log_rates, dtype=float)
    if len(log_rates) < 2:
        raise ValueError(log_rates)

    # subtracting max avoids numerical problems for large |values|
    log_rates -= log_rates.max()
    pdf = numpy.exp(log_rates)

    cdf = numpy.concatenate([[0], numpy.cumsum(pdf)])

    # normalize
    scale = cdf[-1]
    pdf /= scale
    cdf /= scale
    # we don't need the ending 1
    cdf = cdf[:-1]

    args = (start, stop, pdf, cdf)
    return Prior(args, _regular_uniform_between)


@numba.njit
def _regular_uniform_between(args, lo, hi):
    big_lo, smol_lo = _regular_uniform_cdf(args, lo)
    big_hi, smol_hi = _regular_uniform_cdf(args, hi)
    return big_hi - big_lo + (smol_hi - smol_lo)


@numba.njit
def _regular_uniform_cdf(args, x):
    """Return cdf in (big, small) pieces."""
    start, stop, pdf, cdf = args

    if not x < stop:
        return 1.0, 0.0

    if not start < x:
        return 0.0, 0.0

    # regular indexing
    nbins = len(pdf)
    indexf = nbins * (x - start) / (stop - start)

    i = int(indexf)
    frac = indexf - i

    # integrate uniform density in this bin
    box_mass = pdf[i] * frac

    return cdf[i], box_mass
