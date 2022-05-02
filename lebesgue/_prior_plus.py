""" The "plus" prior shifts another prior by a constant. """
import numba

from . import _core
from ._bayes import Prior


def plus(x, prior):
    """Return a prior shifted by x"""
    assert isinstance(prior, Prior)
    x = float(x)
    cls = _plus_class(numba.typeof(self._prior))
    return Prior(cls(x, self._prior))


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
