import numba
import numpy
from numba import f8, i8

from . import _likelihood_poisson, _testing


def test_args():
    assert _testing.raises(lambda: _likelihood_poisson.poisson(None), TypeError)
    assert _testing.raises(lambda: _likelihood_poisson.poisson(-1), ValueError)
    assert _testing.raises(lambda: _likelihood_poisson.poisson(0.5), ValueError)
    assert not _testing.raises(lambda: _likelihood_poisson.poisson(0))
    assert not _testing.raises(lambda: _likelihood_poisson.poisson(1))


def test_poisson_interval():
    rng = numpy.random.Generator(numpy.random.Philox(6))

    xs = rng.normal(size=10)
    ns = [0, 1, 3, 10, 100, 10_000, 1_000_000]

    for n in ns:
        like = _likelihood_poisson.poisson(n)
        mu = n
        sigma = 3 * n**0.5
        for x in xs:
            x_ref = max(0.0, mu + x * sigma + mu)
            ratio = poisson_ratio(n, x_ref)

            lo, hi = like.interval(ratio)

            numpy.testing.assert_allclose(ratio, poisson_ratio(n, lo))
            numpy.testing.assert_allclose(ratio, poisson_ratio(n, hi))


def test_invg_lo():
    xs = numpy.linspace(1e-3, 1, 1000)

    for x_ref in xs:
        y_ref = gfunc(x_ref)

        x_chk = _likelihood_poisson._invg_lo(y_ref)
        y_chk = gfunc(x_chk)

        assert ulp_min(x_ref, y_ref, x_chk, y_chk) <= 2


def test_invg_hi():
    xs = numpy.linspace(1, 1e3, 1000)

    for x_ref in xs:
        y_ref = gfunc(x_ref)

        x_chk = _likelihood_poisson._invg_hi(y_ref)
        y_chk = gfunc(x_chk)

        assert ulp_min(x_ref, y_ref, x_chk, y_chk) <= 1


def test_signatures():
    _likelihood_poisson._poisson_interval(1.0, 1)
    assert _likelihood_poisson._poisson_interval.signatures == [(i8, f8)]


# utilities


def poisson_ratio(n, x):
    assert x >= 0
    assert n >= 0

    if n == 0:
        return numpy.exp(-x)

    if x == 0 or numpy.isinf(x):
        return 0.0

    return numpy.exp(n - x + n * numpy.log(x / n))


def gfunc(x):
    return x - 1 - numpy.log(x)


def ulp_min(x_ref, y_ref, x_chk, y_chk):
    # either close in the function or its inverse
    return min(
        abs(x_ref - x_chk) / numpy.spacing(x_ref),
        abs(y_ref - y_chk) / numpy.spacing(y_ref),
    )
