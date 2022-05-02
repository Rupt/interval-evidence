""" The "plus" prior shifts another prior by a constant. """
import numba

from . import _core
from ._bayes import Prior


def plus(x, prior):
    """Return a prior shifted by x"""
    x = float(x)
    assert isinstance(prior, Prior)
    cls = _plus_class(numba.typeof(prior._prior))
    return Prior(cls(x, prior._prior))


@_core.cache
def _plus_class(prior_type):
    @_core.jitclass
    class _Plus:
        _x: float
        _prior: prior_type

        def __init__(self, x, prior):
            self._x = x
            self._prior = prior

        def _between(self, lo, hi):
            return self._prior._between(lo - self._x, hi - self._x)

    return _Plus
