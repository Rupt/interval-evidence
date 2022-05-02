import itertools
from types import SimpleNamespace

import scipy.integrate

from . import (
    _bayes,
    _core,
    _likelihood_poisson,
    _prior_log_normal,
    _prior_plus,
    _quad_bound,
)


def test_fpow():
    """Check bounds for the simple function fpow."""
    ks = [0, 1, 3, 10]
    los = [0.0, 0.1, 0.4]
    his = [0.5, 0.6, 1.0]

    for k, lo, hi in itertools.product(ks, los, his):
        func, integral = make_fpow(k, lo, hi)

        rtol = 1e-2
        zlo, zhi = _quad_bound._quad_bound(func, rtol=rtol)
        assert zlo <= integral <= zhi
        assert zhi - zlo <= rtol * zlo


def test_model():
    """Test integrating a prior-likelihood model."""
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
