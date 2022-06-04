from test import raises

import numpy

from .likelihood import gamma1, poisson


def test_args():
    assert raises(lambda: poisson(None), TypeError)
    assert raises(lambda: poisson(-1), ValueError)
    assert raises(lambda: poisson(0.5), ValueError)
    assert raises(lambda: poisson(numpy.nan), ValueError)
    assert not raises(lambda: poisson(0))
    assert not raises(lambda: poisson(1))


def test_interval():
    rng = numpy.random.Generator(numpy.random.Philox(6))

    xs = rng.normal(size=10)
    ns = [0, 1, 3, 10, 100, 10_000, 1_000_000]

    for n in ns:
        like = poisson(n)
        mu = n
        sigma = 3 * n**0.5
        for x in xs:
            x_ref = max(0.0, mu + x * sigma + mu)
            ratio = poisson_ratio(n, x_ref)

            lo, hi = like.interval(ratio)

            numpy.testing.assert_allclose(ratio, poisson_ratio(n, lo))
            numpy.testing.assert_allclose(ratio, poisson_ratio(n, hi))


def test_gamma1_args():
    assert raises(lambda: gamma1(None), TypeError)
    assert raises(lambda: gamma1(-1), ValueError)
    assert raises(lambda: gamma1(numpy.nan), ValueError)
    assert not raises(lambda: gamma1(3.2))
    assert not raises(lambda: gamma1(0.2))
    assert not raises(lambda: gamma1(1))


def test_gamma1_interval():
    rng = numpy.random.Generator(numpy.random.Philox(11))

    xs = rng.normal(size=5)
    shapes = [0.5, 1.6, 100.7]

    for shape in shapes:
        like = gamma1(shape)
        mu = shape
        sigma = 3 * shape**0.5
        for x in xs:
            rate_ref = max(0.0, mu + x * sigma + mu)
            ratio = gamma1_ratio(shape, rate_ref)

            lo, hi = like.interval(ratio)

            numpy.testing.assert_allclose(ratio, gamma1_ratio(shape, lo))
            numpy.testing.assert_allclose(ratio, gamma1_ratio(shape, hi))


# utilities


def poisson_ratio(n, x):
    assert n == int(n)
    return gamma1_ratio(n, x)


def gamma1_ratio(shape, rate):
    assert rate >= 0
    assert shape >= 0

    if shape == 0:
        return numpy.exp(-rate)

    if rate == 0 or numpy.isinf(rate):
        return 0.0

    return numpy.exp(shape - rate + shape * numpy.log(rate / shape))
