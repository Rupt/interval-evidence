""" Implement the Poisson likelihood function. """
from collections.abc import Callable

import numba
import numpy
from numba.types import Tuple

from . import _bayes, _core
from ._bayes import Likelihood


def poisson(n):
    """Return a Poisson likelihood for n observed events."""
    if not n == int(n):
        raise ValueError(n)
    n = int(n)
    if not n >= 0:
        raise ValueError(n)
    return Likelihood(n, _poisson_interval)


# inverse order for numba declarations

# root finding iterations


@numba.njit
def _halley_log(x, y):
    u = numpy.log(x)
    # order for case when y ~ 1 so x << 1, u ~ -1
    f = -y - u - 1 + x
    u -= (x - 1) * f / ((x - 1) * (x - 1) - 0.5 * x * f)
    return numpy.exp(u)


@numba.njit
def _halley_lin(x, y):
    f = x - 1 - numpy.log(x) - y
    x -= x * (x - 1) * f / ((x - 1) * (x - 1) - 0.5 * f)
    return x


# low branch


@numba.njit
def _invg_lo_a(y):
    v = numpy.sqrt(y)
    c0 = 1.0
    c1 = 1.4142135623730951
    c2 = 0.6666666666666666
    return c2 * y - c1 * v + c0


@numba.njit
def _invg_lo_b(y):
    a = numpy.exp(-1 - y)
    r = 54 / 5
    r = r * a + 125 / 24
    r = r * a + 8 / 3
    r = r * a + 3 / 2
    r = r * (a * a) + a
    r = r * a + a
    return r


@numba.njit
def _invg_lo_c(y):
    c0 = 1.0
    c1 = 1.4142135623730951
    c2 = 0.6666666666666666
    c3 = -0.07856742094215359
    c4 = -0.014814741391234653
    c5 = -0.0013104253458299017
    c6 = 0.000474467546832846
    c7 = 0.00028305478707334116
    c8 = 7.489873926618484e-05
    c9 = 2.2582275422435765e-05
    c10 = -2.1177244651937083e-05

    v = numpy.sqrt(y)
    r = c10
    r = r * y + c9 * v + c8
    r = r * y + c7 * v + c6
    r = r * y + c5 * v + c4
    r = r * y + c3 * v + c2
    r = r * y - c1 * v + c0
    return r


@numba.njit(cache=True)
def _invg_lo(y):
    if y < 0:
        return numpy.nan

    if y < 2.9103830456733704e-11:
        return _invg_lo_a(y)

    if y > 128:
        return _invg_lo_b(y)

    if y < 2:
        x = _invg_lo_c(y)
    else:
        x = _invg_lo_b(y)

    if y < 0.5:
        return _halley_lin(x, y)

    return _halley_log(x, y)


# high branch


@numba.njit
def _invg_hi_a(y):
    v = numpy.sqrt(y)
    c0 = 1.0
    c1 = 1.4142135623730951
    c2 = 0.6666666666666666
    return c2 * y + c1 * v + c0


@numba.njit
def _invg_hi_b(y):
    # x = -exp(-1 - y)
    # L1 = ln(-x) = -1 - y
    # L2 = ln(-L1) = ln(1 + y) = s
    # s = L2
    # used for y >> 1, so no need for log1p
    s = numpy.log(1 + y)
    c4 = 1 - s * 3 + s * (s * (22 / 12) - s * s * (3 / 12))
    c3 = 1 - s * (9 / 6) + s * s * (2 / 6)
    c2 = 1 - s * (1 / 2)

    a = 1 / (1 + y)
    r = c4
    r = r * a + c3
    r = r * a + c2
    r = r * (a * a) + a
    r = r * s + s + (1 + y)
    return r


@numba.njit
def _invg_hi_c(y):
    c0 = 1.0
    c1 = 1.4142135623730951
    c2 = 0.6666666666666666
    c3 = 0.07856746227849844
    c4 = -0.014816452683579905
    c5 = 0.0013196940772950657
    c6 = 0.0004468042078022748
    c7 = -0.0002648947978101783
    c8 = 6.974109942747075e-05
    c9 = -1.0032032474373403e-05
    c10 = 6.256136117276898e-07

    v = numpy.sqrt(y)
    r = c10
    r = r * y + c9 * v + c8
    r = r * y + c7 * v + c6
    r = r * y + c5 * v + c4
    r = r * y + c3 * v + c2
    r = r * y + c1 * v + c0
    return r


@numba.njit(cache=True)
def _invg_hi(y):
    if y < 0:
        return numpy.nan

    if y < 5.820766091346741e-11:
        return _invg_hi_a(y)

    if y > 8192:
        return _invg_hi_b(y)

    if y < 10.75:
        x = _invg_hi_c(y)
    else:
        x = _invg_hi_b(y)

    return _halley_lin(x, y)


# core


@numba.njit(Tuple([numba.float64, numba.float64])(numba.int64, numba.float64), cache=True)
def _poisson_interval(n, r):
    # log(r) = log(e ** -x * x ** n / (e ** -n * n ** n))
    #        = -n * (x / n - 1 - log(x / n))
    # => -log(r) / n = (x / n) - 1 - log(x / n) = g(x / n)
    # => x = n * invg(-log(r) / n)
    if n < 0:
        return numpy.nan, numpy.nan

    if n == 0 or r == 0.0:
        # r = e^-x => x = -log(r)
        return 0.0, -numpy.log(r)

    y = -numpy.log(r) / n
    return n * _invg_lo(y), n * _invg_hi(y)
