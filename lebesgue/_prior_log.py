""" Transform a prior to apply in log space. """
import functools

import numba
import numpy

from ._bayes import Prior


def log(prior: Prior) -> Prior:
    """Return a Prior for `prior' applied in log space.

    Arguments:
        prior: another Prior object to transform
    """
    between_func = _log_between(prior.between_func)
    return Prior(prior.args, between_func)


@functools.lru_cache(maxsize=None)
def _log_between(between_func):
    @numba.njit
    def between(args, lo, hi):
        if lo > 0:
            lo_new = numpy.log(lo)
        else:
            lo_new = -numpy.inf

        if hi > 0:
            hi_new = numpy.log(hi)
        else:
            hi_new = -numpy.inf

        return between_func(args, lo_new, hi_new)

    return between
