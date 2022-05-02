import itertools
from types import SimpleNamespace

from . import _quad_bound


def test():
    test_fpow()


def test_fpow():
    """Check bounds for the simple function fpow."""
    ks = [0, 1, 3, 10, 1000]
    los = [0.0, 0.1, 0.4]
    his = [0.5, 0.6, 1.0]

    for k, lo, hi in itertools.product(ks, los, his):
        func, integral = make_fpow(k, lo, hi)
        zlo, zhi = wrap_quad_bound(func, rtol=1e-2)
        assert zlo <= integral <= zhi


def wrap_quad_bound(func, *, rtol):
    obj = SimpleNamespace(_mass=func)
    return _quad_bound._quad_bound(obj, rtol)


def make_fpow(k, lo, hi):
    assert lo < hi, (lo, hi)

    # rescale such that 0 -> lo, 1 -> hi
    def fpow(x):
        x = lo + (hi - lo) * x
        return (1 - x) ** k

    integral = ((1 - lo) ** (k + 1) - (1 - hi) ** (k + 1)) / (k + 1)
    # jacobian factor for the rescaling
    integral /= hi - lo

    return fpow, integral
