""" Truncate priors to their segments between two limits. """
import functools

import numba

from ._bayes import _Prior


def trunc(lo: float, hi: float, prior: _Prior) -> _Prior:
    """Return prior truncated between lo and hi."""
    lo = float(lo)
    hi = float(hi)
    if not hi > lo:
        raise ValueError((lo, hi))

    args = (lo, hi, prior.args)
    between_func = _trunc_between(prior.between_func)
    return _Prior(args, between_func)


@functools.lru_cache(maxsize=None)
def _trunc_between(between_func):
    @numba.njit
    def between(args, lo, hi):
        trunc_lo, trunc_hi, args_inner = args

        norm = between_func(trunc_lo, trunc_hi)

        # beware div0 errors
        return between_func(max(lo, trunc_lo), min(hi, trunc_hi)) / norm

    return between
