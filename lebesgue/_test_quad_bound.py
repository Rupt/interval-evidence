import itertools
from types import SimpleNamespace

from . import _quad_bound


def test():
    test_fpow()


def test_fpow():
    """Check bounds for the simple function fpow."""
    ns = [0, 1, 3, 10, 1000]
    los = [0.0, 0.1, 0.4]
    his = [0.4, 0.6, 1.0]

    for n, lo, hi in itertools.product(ns, los, his):
        func, integral = make_fpow(3, 0.25, 0.5)
        zlo, zhi = wrap_quad_bound(func, rtol=1e-2)
        assert zlo <= integral <= zhi


def wrap_quad_bound(func, *, rtol):
    obj = SimpleNamespace(_mass=func)
    return _quad_bound._quad_bound(obj, rtol)


def make_fpow(n, lo, hi):
    assert lo <= hi, (lo, hi)

    # rescale such that 0 -> lo, 1 -> hi
    def fpow(x):
        x = lo + (hi - lo) * x
        return (1 - x) ** n

    integral = ((1 - lo) ** (n + 1) - (1 - hi) ** (n + 1)) / (n + 1)
    # jacobian factor for the rescaling
    integral /= hi - lo

    return fpow, integral
