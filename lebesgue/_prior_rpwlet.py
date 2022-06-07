"""A `regular piece-wise linear exponentially tailed' (rpwlet) prior.

Pronounced "Rowlet", as in "pwn":

                         rrrrrrrrrrrrrrrrrrrrrr
                    rrrrrrrrr               rrrrrrrr
                rrrrrrr                r          rrrrr
              rrrrrrrrrrrrrrrr   rrrrrrrrrrrrrrrr    rrrrr
            rrrrrr           rrrrr             rrrrr    rrrr
          rrrr              rrrrrr                 rrr    rrr
         rrr              rrr    rrr                 rr     rrr
       rrrr     rrrrrrr  rr        rr   rrrrrrr       rr     rrr
       rrr     rrrrrrrrrrr         rr  rrrrrrrrr       rr      rr
      rrrr     rrrrrrrrrrr         rr  rrrrrrrrr       rr      rrr
     rrrrr     rrrrrrrrrrr         rr  rrrrrrrrr       rr       rr
     rr rr     rrrrrrrrrrr  r     rrr  rrrrrrrrr       rr        rr
    rr  rrr     rrrrrrr rrrrrrrrrrrrr   rrrrrrr       rr         rr
    rr   rrr       r     rrrrr   rrr       r         rr           rr
    rr     rr              rrrrrrr                 rrr            rr
    rr      rrrrr           rrrrrr              rrrrr             rr
    rr         rrrrrrrrrrrr r    rrrrrrrrrrrrrrrrr                rr
    rr              rrrrrr         rrrrrrrrr                      rr
     rr          rrrrrrrrrrrrr  rrrrrrr rrrrrrr                  rrr
      r        rrr           rrrr            rrrr                rr
      rr  r   rr          rrrrrr rrrrr         rrr    rr        rrr
       rrrrr  rrr             rrr            rrrr    rrr        rr
        rrrrr   rrrrrr     rrrrrrrr      rrrrrrr   rrrr       rrr
          rrrrrrrr rrrrrrrrrrr   rrrr rrrrrr   rrrrrrrr    r rrr
           rrrrrr                               rr   rr  rrrrr
             rrrrrr                                  rrrrrrr
            rrrrrr                              rrrrr
                     rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr
                rr   rr  rrrrrrrrrrr   rrrrrrrrr    rrr
               rr  rrrr   rrrrr         rr   rrrrr   rr
               rrrrrrrrrrrrr            rrrrrrrrrrrrrrr
                     rrrr                rrr       rrr

    -owlet!
"""
from collections.abc import Sequence

import numba
import numpy

from ._bayes import Prior


def rpwlet(log_rates: Sequence[float], stop: float) -> Prior:
    """Return a Prior with piecewise linear denisty up to stop, and an
    exponential tail thereafter.

    The exponential tail is chosen for a density which is continuous and
    continuous in its first derivative.

    Arguments:
        log_rates: log density + c, at regularly spaced intervals
            located at numpy.linspace(0, stop, len(rates))
    """
    if not stop == float(stop):
        raise ValueError(stop)
    stop = float(stop)
    if not stop > 0:
        raise ValueError(stop)

    log_rates = numpy.asarray(log_rates, dtype=float)
    # subtracting max avoids numerical problems for large |values|
    log_rates = log_rates - log_rates.max()
    rates = numpy.exp(log_rates)

    # linear bulk
    delta = stop / (len(rates) - 1)
    masses = 0.5 * (rates[1:] + rates[:-1]) * delta

    # exponential tail
    # match gradient, scale, and location
    # (1) f = exp(a - b x); f(stop) = rate(stop)
    # (2) f' = -b f; f'(stop) = d/dx(rate(stop))
    log_rate_stop = log_rates[-1]
    rate_stop = rates[-1]
    rate_dash_stop = (rates[-2] - rate_stop) / delta

    # (2) => b = -d/dx(rate(stop)) / rate(stop)
    tail_b = rate_dash_stop / rate_stop
    if not tail_b > 0:
        raise ValueError(tail_b)

    # (1) => a = log(rate(stop)) + b x
    tail_a = log_rate_stop + tail_b * stop

    # integrate exp(a - b * x) from end to inf
    # a - b * stop = log(rate_stop)
    tail_mass = rate_stop / tail_b

    # normalize and reparameterize
    cmf = numpy.concatenate([[0], numpy.cumsum(masses)])
    norm = 1 / (cmf[-1] + tail_mass)

    cmf_norm = cmf * norm
    pdf_norm = rates * norm

    tail_c = tail_a + numpy.log(norm / tail_b)
    tail_e = cmf_norm[-1] + numpy.exp(tail_c - tail_b * stop)

    args = (stop, pdf_norm, cmf_norm, tail_b, tail_c, tail_e)

    def cdf(x):
        nboxes = len(pdf_norm) - 1

        delta = stop / nboxes
        indexf = x / delta

        # tail integral
        if not indexf < nboxes:
            return tail_e - numpy.exp(tail_c - tail_b * x)

        # piecewise linear
        i = int(indexf)
        frac = indexf - i

        # integrate density in this box, where density linearly increases
        # pdf = pdf_lo + (pdf_hi - pdf_lo) * frac
        pdf_lo = pdf_norm[i]
        pdf_hi = pdf_norm[i + 1]
        box_mass = delta * frac * (pdf_lo + 0.5 * (pdf_hi - pdf_lo) * frac)

        return cmf_norm[i] + box_mass

    for x in [0, 1, 2, 10, 20, 21]:
        print(cdf(x))
        print(_rpwlet_between(args, 0, x))
        print(_rpwlet_between(args, 1, x))


@numba.njit
def _rpwlet_cdf(args, x):
    """Return cdf in (big, small) pieces."""
    stop, pdf_norm, cmf_norm, tail_b, tail_c, tail_e = args

    nboxes = len(pdf_norm) - 1

    delta = stop / nboxes
    indexf = x / delta

    # tail integral
    if not indexf < nboxes:
        return tail_e, -numpy.exp(tail_c - tail_b * x)

    # piecewise linear
    i = int(indexf)
    frac = indexf - i

    # integrate density in this box, where density linearly increases
    # pdf = pdf_lo + (pdf_hi - pdf_lo) * frac
    pdf_lo = pdf_norm[i]
    pdf_hi = pdf_norm[i + 1]
    box_mass = delta * frac * (pdf_lo + 0.5 * (pdf_hi - pdf_lo) * frac)

    return cmf_norm[i], box_mass


@numba.njit
def _rpwlet_between(args, lo, hi):
    big_lo, small_lo = _rpwlet_cdf(args, lo)
    big_hi, small_hi = _rpwlet_cdf(args, hi)
    return (big_hi - big_lo) + (small_hi - small_lo)
