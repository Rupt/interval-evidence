import numba
import numpy

from . import Model, likelihood, prior
from ._quad_bound import _next_pow2, integrator


def test_fpow():
    """Check bounds for the simple function fpow."""

    # very large k (>~1e20) can lead to numerical failure in fpow
    k_lo_his = [
        (0, 0, 1),
        (10, 0.1, 0.9),
        (100, 0, 0.5),
        (1e15, 0, 0.8),
    ]
    for k, lo, hi in k_lo_his:
        args, integral = make_fpow(k, lo, hi)

        rtol = 1e-2
        zlo, zhi = integrator(fpow)(args, rtol=rtol)
        assert zlo <= integral <= zhi
        assert zhi - zlo <= rtol * zlo


def test_normal():
    """Check against the analytic result for normal-normal models."""
    mu1_sigma1_mu2_sigma2 = [
        (0, 1, 0, 1),
        (0, 1, 1, 1),
        (0, 1, 1, 2),
        (-2, 0.2, 3, 0.1),
    ]

    for args in mu1_sigma1_mu2_sigma2:
        integral_ref = numpy.exp(normal_normal_logz_ratio(*args))
        lo, hi = normal_normal(*args).integrate()
        # reference has far higher precision
        assert lo <= integral_ref <= hi


def test_next_pow2():
    assert _next_pow2(0.001) == 1
    assert _next_pow2(1) == 1
    assert _next_pow2(1.001) == 2
    assert _next_pow2(512) == 512
    assert _next_pow2(512.1) == 1024
    assert _next_pow2(2**52) == 2**52
    assert _next_pow2(2**52 + 1) == 2**53
    assert numpy.isnan(_next_pow2(numpy.nan))


# utilities


def make_fpow(k, lo, hi):
    assert lo < hi, (lo, hi)

    args = (float(k), float(lo), float(hi))

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


_integrate_fpow = integrator(fpow)


@integrator.put(fpow)
@numba.njit(cache=True)
def integrate_fpow(args, rtol):
    return _integrate_fpow(args, rtol)


# normal-normal references


def normal_normal(mu1, sigma1, mu2, sigma2):
    return Model(likelihood.normal(mu1, sigma1), prior.normal(mu2, sigma2))


_model = normal_normal(0, 1, 0, 1)

_normal_normal_integrate_func = _model.integrate_func


@integrator.put(_model.mass_func)
@numba.njit(cache=True)
def _normal_normal_integrate(args, rtol):
    return _normal_normal_integrate_func(args, rtol)


def normal_normal_logz_ratio(mu1, sigma1, mu2, sigma2):
    occam_term = numpy.log1p(sigma2**2 / sigma1**2)
    max_term = (mu1 - mu2) ** 2 / (sigma1**2 + sigma2**2)
    return -0.5 * (occam_term + max_term)
