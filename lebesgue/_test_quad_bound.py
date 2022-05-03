import itertools

import numba

from . import _quad_bound


def test_fpow():
    """Check bounds for the simple function fpow."""
    ks = [0, 1, 3, 10]
    los = [0.0, 0.1, 0.4]
    his = [0.5, 0.6, 1.0]

    for k, lo, hi in itertools.product(ks, los, his):
        args, integral = make_fpow(k, lo, hi)

        rtol = 1e-2
        zlo, zhi = quad_bound_fpow(args, rtol=rtol)
        assert zlo <= integral <= zhi
        assert zhi - zlo <= rtol * zlo


# utilities


def make_fpow(k, lo, hi):
    assert lo < hi, (lo, hi)

    args = (int(k), float(lo), float(hi))

    integral = ((1 - lo) ** (k + 1) - (1 - hi) ** (k + 1)) / (k + 1)
    # jacobian factor for the rescaling
    integral /= hi - lo

    return args, integral


@numba.njit
def fpow(args, x):
    k, lo, hi = args
    # resale such that 0 -> lo, 1 -> hi
    x = lo + (hi - lo) * x
    return (1 - x) ** k


_quad_bound_fpow = _quad_bound.generate(fpow)


@numba.njit(cache=True)
def quad_bound_fpow(args, rtol):
    return _quad_bound_fpow(args, rtol)
