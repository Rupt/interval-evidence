from test import raises

import numpy

from .prior import rpwlet


def test_first_example():
    # compare against an initial example
    log_rates = [
        -85.48754627,
        -83.33749391,
        -81.55292162,
        -81.14965234,
        -81.13650134,
        -81.30486231,
        -81.57363637,
        -81.90427197,
        -82.27577931,
        -82.67559802,
        -83.09568261,
        -83.53061175,
        -83.97658626,
        -84.43085962,
        -84.89139544,
        -85.35665259,
        -85.82544405,
        -86.29684206,
        -86.77011358,
        -87.24467272,
        -87.72004939,
    ]
    stop = 19.1
    ref = [
        0.0,
        0.3111276812278084,
        0.79351272830663,
        0.9542508729629473,
        0.9909604739200537,
        0.9983588293623089,
        0.9997730780216482,
        0.9999728798187854,
        0.9999967587792312,
        0.9999996126311993,
    ]

    prior = rpwlet(log_rates, stop)

    numpy.testing.assert_allclose(
        ref, [prior.between(0, x) for x in numpy.linspace(0, 30, 10)]
    )


def test_values():
    rates = [2, 1]
    stop = 1

    prior = rpwlet(numpy.log(rates), stop)

    assert prior.between(0, 0) == 0
    assert prior.between(0, 1e300) == 1

    eps = 1e-6

    def density(x):
        return prior.between(x, x + eps) / eps

    # check piecewise linearity of density
    norm = density(0) / rates[0]
    assert numpy.isclose(density(stop), rates[1] * norm)
    assert numpy.isclose(
        density(0.5 * stop), 0.5 * (rates[0] + rates[1]) * norm
    )

    # check exponential decay
    assert numpy.isclose(density(stop + 1), density(stop + 2) * numpy.e)

    # check continuous derivative
    def derivative(x):
        return (density(x + eps) - density(x)) / eps

    assert numpy.isclose(derivative(stop - 2 * eps), derivative(stop + eps))


def test_args():
    assert raises(lambda: rpwlet([1, 2], 0), ValueError)
    assert raises(lambda: rpwlet([1, 2], -1), ValueError)
    assert raises(lambda: rpwlet([1, 2], None), TypeError)
    assert raises(lambda: rpwlet([1], 1), ValueError)
    # non-decreasing tail
    assert raises(lambda: rpwlet([1, 1], 1), ValueError)
    assert raises(lambda: rpwlet([1, 2], 1), ValueError)
