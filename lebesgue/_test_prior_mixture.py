import warnings
from test import raises

import numba

from .prior import mixture, normal


def test_values_pair():
    prior_a = normal(1, 1)
    prior_b = normal(4, 1)

    rate_a = 0.5
    rate_b = 1.5

    lo = 2
    hi = 3

    between_a = prior_a.between(lo, hi)
    between_b = prior_a.between(lo, hi)

    norm = 1 / (rate_a + rate_b)

    between_ref = between_a * (rate_a * norm) + between_b * (rate_b * norm)

    between_check = mixture(
        [
            (prior_a, rate_a),
            (prior_b, rate_b),
        ]
    ).between(lo, hi)

    assert between_ref == between_check


def test_values_trio():
    priors = (
        normal(-1, 0.5),
        normal(0, 0.5),
        normal(2, 0.5),
    )

    rates = range(3)

    norm = 1 / sum(rates)

    lo = -0.5
    hi = 1.5

    between_check = 0.0
    for prior, rate in zip(priors, rates):
        between_check += prior.between(lo, hi) * (rate * norm)

    # numba function support is experimental and yells about it
    with warnings.catch_warnings():
        category = numba.NumbaExperimentalFeatureWarning
        warnings.filterwarnings("ignore", category=category)

        between_ref = mixture(zip(priors, rates)).between(lo, hi)

    assert between_ref == between_check


def test_args():
    assert raises(lambda: mixture([]), ValueError)
    assert raises(lambda: mixture([(normal(0, 1), 0.0)]), ValueError)
    assert raises(
        lambda: mixture(
            [
                (normal(0, 1), 3.1),
                (normal(0, 1), None),
            ]
        ),
        TypeError,
    )
    assert raises(
        lambda: mixture(
            [
                (None, 1),
                (normal(0, 1), 1),
            ]
        ),
        AttributeError,
    )
