""" Shift a prior by a constant. """
import functools

import numba

from ._bayes import Prior


def add(x: float, prior: Prior) -> Prior:
    """Return a Prior for `prior' shifted by x.

    Arguments:
        x: shift amount
        prior: another Prior object to transform
    """
    x = float(x)
    args = (x, prior.args)
    between_func = _add_between(prior.between_func)
    return Prior(args, between_func)


@functools.lru_cache(maxsize=None)
def _add_between(between_func):
    @numba.njit
    def between(args, lo, hi):
        x, args_inner = args
        return between_func(args_inner, lo - x, hi - x)

    return between
