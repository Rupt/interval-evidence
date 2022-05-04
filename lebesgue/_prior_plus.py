""" The "plus" prior shifts another prior by a constant. """
import functools

import numba

from ._bayes import _Prior


def plus(x: float, prior: _Prior) -> _Prior:
    """Return a Prior for `prior' shifted by x.

    Arguments:
        x: shift amount
        prior: another _Prior object to transform
    """
    x = float(x)
    if not isinstance(prior, _Prior):
        raise TypeError(prior)

    args = (x, prior.args)
    between_func = _plus_between(prior.between_func)

    return _Prior(args, between_func)


# caching reduces recompilation, which is expecsive


@functools.lru_cache(maxsize=None)
def _plus_between(between_func):
    @numba.njit
    def _between(args, lo, hi):
        x, args_inner = args
        return between_func(args_inner, lo - x, hi - x)

    return _between
