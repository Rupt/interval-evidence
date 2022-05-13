import itertools

import numba
import numpy

from . import Model, likelihood, prior
from ._quad_bound import integrator


def test_fpow():
    """Check bounds for the simple function fpow."""
    ks = [0, 1, 10]
    los = [0.0, 0.4]
    his = [0.5, 1.0]

    for k, lo, hi in itertools.product(ks, los, his):
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
def _normal_normal_integrate(args, ratio):
    return _normal_normal_integrate_func(args, ratio)


def normal_normal_logz_ratio(mu1, sigma1, mu2, sigma2):
    occam_term = numpy.log1p(sigma2**2 / sigma1**2)
    max_term = (mu1 - mu2) ** 2 / (sigma1**2 + sigma2**2)
    return -0.5 * (occam_term + max_term)
