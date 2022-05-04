""" Truncate priors to their segments between two limits. """
import functools

import numba

from ._bayes import _Prior


def trunc(lo: float, hi: float, prior: _Prior) -> _Prior:
    """
    Return prior truncated and normalized between lo and hi.

    Arguments:
        lo: lower bound
        hi: upper bound (> lo)
        prior: inner prior to modify

    """
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
        lo_trunc, hi_trunc, args_inner = args

        norm = between_func(args_inner, lo_trunc, hi_trunc)

        # beware div0 errors
        return (
            between_func(args_inner, max(lo, lo_trunc), min(hi, hi_trunc))
            / norm
        )

    return between
