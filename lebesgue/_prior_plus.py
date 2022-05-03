""" The "plus" prior shifts another prior by a constant. """
import functools

import numba

from ._bayes import Prior


def plus(x, prior):
    """Return a prior shifted by x"""
    x = float(x)
    if not isinstance(prior, Prior):
        raise TypeError(prior)

    args = (x, prior._args)
    between_func = _plus_between(prior._between_func)

    return Prior(args, between_func)


@functools.lru_cache(maxsize=None)
def _plus_between(between_func):
    @numba.njit
    def _between(args, lo, hi):
        x, args_inner = args
        return between_func(args_inner, lo - x, hi - x)

    return _between
