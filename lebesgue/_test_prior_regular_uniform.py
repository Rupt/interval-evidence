from test import raises

import numpy

from ._prior_regular_uniform import regular_uniform


def test_values():
    rates = [1, 0.5, 1.5, 3]
    start = 0.5
    stop = 2

    prior = regular_uniform(start, stop, numpy.log(rates))

    assert prior.between(-numpy.inf, start) == 0
    assert prior.between(-numpy.inf, stop) == 1
    assert prior.between(-numpy.inf, numpy.inf) == 1

    # check piecewise linearity
    linspace = numpy.linspace(start, stop, len(rates) + 1)
    for i in range(len(rates)):
        begin = linspace[i]
        end = linspace[i + 1]

        mass = prior.between(begin, end)

        for frac in [0.1, 0.33, 0.5, 0.8]:
            assert numpy.isclose(
                mass * frac, prior.between(begin, begin + frac * (end - begin))
            )


def test_args():
    assert raises(lambda: regular_uniform(0, 0, [1, 2]), ValueError)
    assert raises(lambda: regular_uniform(0, -1, [1, 2]), ValueError)
    assert raises(lambda: regular_uniform(0, 1, []), ValueError)
    assert raises(lambda: regular_uniform(numpy.nan, 1, [1, 2]), ValueError)
