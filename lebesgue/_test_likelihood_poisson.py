import numpy

from ._test import raises
from .likelihood import poisson


def test_args():
    assert raises(lambda: poisson(None), TypeError)
    assert raises(lambda: poisson(-1), ValueError)
    assert raises(lambda: poisson(0.5), ValueError)
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


# utilities


def poisson_ratio(n, x):
    assert x >= 0
    assert n >= 0

    if n == 0:
        return numpy.exp(-x)

    if x == 0 or numpy.isinf(x):
        return 0.0

    return numpy.exp(n - x + n * numpy.log(x / n))
