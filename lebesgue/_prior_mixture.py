"""Implement an additive combination of priors."""
import functools
from collections.abc import Sequence

import numba
import numpy

from ._bayes import Prior


def mixture(prior_and_rates: Sequence[Prior, float]) -> Prior:
    """Return a Prior of priors weighted in proportion to rates.

    Arguments:
        prior_and_rates: pairs of Prior with a rate proportional to its weight;
            must have length 2 or more

    """
    pairs = list(prior_and_rates)

    if len(pairs) < 2:
        raise ValueError(pairs)

    between_funcs = tuple(prior.between_func for prior, _ in pairs)
    prior_args = tuple(prior.args for prior, _ in pairs)
    rates = [rate for _, rate in pairs]

    norm = 1 / numpy.sum(rates)
    weights = tuple(rate * norm for rate in rates)

    args = (weights, prior_args)
    between_func = _mixture_between(between_funcs, weights)

    return Prior(args, between_func)


@functools.lru_cache(maxsize=None)
def _mixture_between(funcs, weights):
    if len(funcs) == 2:
        # numba function support is experimental, so avoid using them in tuple
        # for small mixtures
        func_0 = funcs[0]
        func_1 = funcs[1]

        @numba.njit
        def between(args, lo, hi):
            weights, prior_args = args

            return (
                func_0(prior_args[0], lo, hi) * weights[0]
                + func_1(prior_args[1], lo, hi) * weights[1]
            )

    else:

        @numba.njit
        def between(args, lo, hi):
            weights, prior_args = args

            value = 0.0
            for func, args, weight in zip(funcs, prior_args, weights):
                value += func(args, lo, hi) * weight

            return value

    return between
