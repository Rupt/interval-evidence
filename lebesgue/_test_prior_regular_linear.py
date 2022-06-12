from test import raises

import numpy

from ._prior_regular_linear import regular_linear


def test_values():
    rates = [1, 0.5, 1.5, 3]
    start = 0.5
    stop = 2

    prior = regular_linear(start, stop, numpy.log(rates))

    assert prior.between(-numpy.inf, start) == 0
    assert prior.between(-numpy.inf, stop) == 1
    assert prior.between(-numpy.inf, numpy.inf) == 1

    eps = 1e-6

    def density(x):
        return prior.between(x, x + eps) / eps

    # check piecewise linearity of density
    norm = density(start) / rates[0]
    linspace = numpy.linspace(start, stop, len(rates))
    for i in range(1, len(rates) - 1):
        # last one flattens after; check just before
        assert numpy.isclose(density(linspace[i] - eps), rates[i] * norm)


def test_args():
    assert raises(lambda: regular_linear(0, 0, [1, 2]), ValueError)
    assert raises(lambda: regular_linear(0, -1, [1, 2]), ValueError)
    assert raises(lambda: regular_linear(0, 1, [1]), ValueError)
    assert raises(lambda: regular_linear(numpy.nan, 1, [1, 2]), ValueError)
