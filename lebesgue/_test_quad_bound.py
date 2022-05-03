import itertools
from types import SimpleNamespace

import scipy.integrate

from . import (  # _bayes,; _likelihood_poisson,; _prior_log_normal,; _prior_plus,
    _core,
    _quad_bound,
)


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


def test_model():
    """Test integrating a Model, which uses _quad_bound."""
    ns = [0, 3, 10]
    shifts = [0, 2]
    means = [0, 1, 5]

    for n, shift, mean in itertools.product(ns, shifts, means):
        model = _bayes.Model(
            _likelihood_poisson.poisson(n),
            _prior_plus.plus(
                shift,
                _prior_log_normal.log_normal(mean, 1.0),
            ),
        )

        rtol = 1e-2
        zlo, zhi = model.integrate(rtol=rtol)
        assert zhi - zlo <= rtol * zlo

        # quad comes with an absolute error estimate; be generous with it
        chk, chk_err = scipy.integrate.quad(model.mass, 0, 1, epsabs=0, epsrel=1e-8)
        assert zlo <= chk + chk_err
        assert zhi >= chk - chk_err


# utilities


def make_fpow(k, lo, hi):
    assert lo < hi, (lo, hi)

    args = (int(k), float(lo), float(hi))

    integral = ((1 - lo) ** (k + 1) - (1 - hi) ** (k + 1)) / (k + 1)
    # jacobian factor for the rescaling
    integral /= hi - lo

    return args, integral


@_core.jit
def fpow(args, x):
    k, lo, hi = args
    # resale such that 0 -> lo, 1 -> hi
    x = lo + (hi - lo) * x
    return (1 - x) ** k


_quad_bound_fpow = _quad_bound.generate(fpow)


@_core.jit(cache=True)
def quad_bound_fpow(args, rtol):
    return _quad_bound_fpow(args, rtol)
