"""
Poisson/Gamma interval implementation.

We evaluate the two real inverses by choices from polynomial approximations
each followed by one iteration of Halley's root-finding method.


The poisson likelihood is exp(-x) * x ** n / n!, so its log ratio from maximum

    log(f(x) / max f(x)) = -n * (x / n - 1 - log(x / n))                    (1)
                         = -n * g(x / n) ,                                  (2)

where

    g(x) = x - 1 - log(x) .                                                 (3)

We find the interval above a given ratio by inverting g and mopping up n.

Some properties of g are:

    g >= 0,

    unique minimum at g(x=1) == 0,

    g(x->0) -> +inf,

    g(x->inf) -> +inf .


The Lambert W function is defined such that

    x = W_k(y)    =>    y = x exp(x) .                                      (4)

W_k is real solutions when k = 0 or k = -1.

The inverses of g are given by

    invg(y) = -W_k(-exp(-1 - y))                                            (5)

since

    -exp(-1 - y) = x exp(x)    =>    y = (-x) - 1 - log(-x) .               (6)

"On the LambertW function" doi:10.1007/BF02124750 gives asymptotic expansions
in L1 = ln(exp(-1 - y)) = -1 - y and L2 = ln(-L1) (reproduced on Wikipedia).

k = 0 corresponds to the low 'lo' half with x <= 1.
k = -1 corresponds to the high 'hi' half with x >= 1.


Our implementation is as follows:

lo, small y, use invg(y) = 1 - sqrt(2) * sqrt(y) + (2 / 3) * y .            (7)

lo, large y, use the Taylor expansion of -W_0(-x) at 0.

lo, intermediate y, use a fitted polynomial in sqrt(y) leading with (7).


hi, small y, use invg(y) = 1 + sqrt(2) * sqrt(y) + (2 / 3) * y.             (8)

hi, large y, use the asymptotic expansion.

hi, intermediate y, use a fitted polynomial in sqrt(y) leading with (8).


We consider the solution accurate if either x = invg(g(x)) or y = g(invg(y)).
This is appropriate since the extreme gradients of g do not always allow exact
evaluations or inverses.

Approximating the units-(in)-last-place distance as

    ulp(x, x_approx) = abs(x_approx - x) / spacing(x) ,                     (9)

we define a distance function

    D(x, x_approx, y, y_approx) = min(ulp(x, x_approx), ulp(y, y_approx))  (10)

to allow either to be accurate. Numerical experiments find D <= 2 for the lo
half and D <= 1 for the hi half.

"""
import numba
import numpy

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


@numba.njit(numba.typeof((0.0, 0.0))(numba.float64, numba.float64), cache=True)
def ginterval(shape, ratio):
    """Return lo, hi within which -shape * g(x / shape) == log(ratio).

    For the poisson case, n is an integer and shape is n.
    """
    if not shape >= 0 or not 0 <= ratio <= 1:
        return numpy.nan, numpy.nan

    if ratio == 0:
        return 0.0, numpy.inf

    if shape == 0:
        # r = e^-x => x = -log(r)
        return 0.0, -numpy.log(ratio)

    y = -numpy.log(ratio) / shape
    return shape * _invg_lo(y), shape * _invg_hi(y)
