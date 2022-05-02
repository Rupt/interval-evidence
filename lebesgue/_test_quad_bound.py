import itertools
from types import SimpleNamespace

from . import _core, _quad_bound


def test_fpow():
    """Check bounds for the simple function fpow."""
    ks = [0, 1, 3, 10, 1000]
    los = [0.0, 0.1, 0.4]
    his = [0.5, 0.6, 1.0]

    for k, lo, hi in itertools.product(ks, los, his):
        func, integral = make_fpow(k, lo, hi)
        zlo, zhi = _quad_bound._quad_bound(func, rtol=1e-2)
        assert zlo <= integral <= zhi


def make_fpow(k, lo, hi):
    assert lo < hi, (lo, hi)

    func = Fpow(int(k), float(lo), float(hi))

    integral = ((1 - lo) ** (k + 1) - (1 - hi) ** (k + 1)) / (k + 1)
    # jacobian factor for the rescaling
    integral /= hi - lo

    return func, integral


@_core.jitclass
class Fpow:
    _k: int
    _lo: float
    _hi: float

    def __init__(self, k, lo, hi):
        self._k = k
        self._lo = lo
        self._hi = hi

    def _mass(self, x):
        # resale such that 0 -> lo, 1 -> hi
        x = self._lo + (self._hi - self._lo) * x
        return (1 - x) ** self._k
