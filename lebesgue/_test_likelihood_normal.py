import itertools

import numpy

from ._likelihood_normal import normal
from ._test import raises


def test_args():
    assert raises(lambda: normal(None, 1), TypeError)
    assert raises(lambda: normal(1, None), TypeError)
    assert raises(lambda: normal(0.0, 0.0), ValueError)
    assert raises(lambda: normal(0.0, -1.0), ValueError)
    assert not raises(lambda: normal(3.0, 1.0))


def test_interval():
    mu_sigmas = [
        (0.0, 1.0),
        (1.0, 1.0),
        (1.0, 2.0),
    ]

    ratios = numpy.linspace(0.0, 1.0, 5)

    for (mu, sigma), ratio in itertools.product(mu_sigmas, ratios):
        lo, hi = normal(mu, sigma).interval(ratio)
        assert numpy.allclose(normal_likelihood_ratio(mu, sigma, lo), ratio)
        assert numpy.allclose(normal_likelihood_ratio(mu, sigma, hi), ratio)


def normal_likelihood_ratio(mu, sigma, x):
    chi = (x - mu) / sigma
    return numpy.exp(-0.5 * chi**2)
