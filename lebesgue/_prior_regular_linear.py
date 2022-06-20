"""A regularly piece-wise linear prior."""
import numba
import numpy

from ._bayes import Prior


def regular_linear(start: float, stop: float, log_rates: list[float]) -> Prior:
    """Return a Prior with piecewise linear denisty from start to stop.

    Arguments:
        start: low edge
        stop: high edge
        log_rates: log density [+ c] at each edge

            Edges are located at numpy.linspace(start, stop, len(log_rates))
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

    masses = 0.5 * (pdf[1:] + pdf[:-1])
    cdf = numpy.concatenate([[0], numpy.cumsum(masses)])

    # normalize
    scale = cdf[-1]
    cdf /= scale
    pdf /= scale

    args = (start, stop, pdf, cdf)
    return Prior(args, _regular_linear_between)


@numba.njit
def _regular_linear_between(args, lo, hi):
    big_lo, smol_lo = _regular_linear_cdf(args, lo)
    big_hi, smol_hi = _regular_linear_cdf(args, hi)
    return big_hi - big_lo + (smol_hi - smol_lo)


@numba.njit
def _regular_linear_cdf(args, x):
    """Return cdf in (big, small) pieces."""
    start, stop, pdf, cdf = args

    if not x < stop:
        return 1.0, 0.0

    if not start < x:
        return 0.0, 0.0

    # regular indexing
    nbins = len(pdf) - 1
    indexf = nbins * (x - start) / (stop - start)

    i = int(indexf)
    frac = indexf - i

    # integrate density in this box, where density linearly increases
    # pdf = pdf_lo + (pdf_hi - pdf_lo) * frac
    pdf_lo = pdf[i]
    pdf_hi = pdf[i + 1]
    box_mass = frac * (pdf_lo + 0.5 * (pdf_hi - pdf_lo) * frac)

    return cdf[i], box_mass
