"""Define tests for full bayes models."""
import itertools

import scipy.integrate

from . import _bayes, _likelihood_poisson, _prior_log_normal, _prior_plus


def test_poisson_log_normal():
    ns = [1, 2]
    mus = [-1, 2]
    sigmas = [0.5, 1.5]

    for n, mu, sigma in itertools.product(ns, mus, sigmas):
        model = _bayes.Model(
            _likelihood_poisson.poisson(n),
            _prior_log_normal.log_normal(mu, sigma),
        )

        rtol = 1e-2
        zlo, zhi = model.integrate(rtol=rtol)
        assert zhi - zlo <= rtol * zlo

        # quad comes with an absolute error estimate; be generous with it
        chk, chk_err = scipy.integrate.quad(
            model.mass, 0, 1, epsabs=0, epsrel=1e-8
        )
        assert zlo <= chk + chk_err
        assert zhi >= chk - chk_err


def test_poisson_plus_log_normal():
    ns = [0, 3]
    shifts = [0, 2]
    mus = [0, 5]

    for n, shift, mu in itertools.product(ns, shifts, mus):
        model = _bayes.Model(
            _likelihood_poisson.poisson(n),
            _prior_plus.plus(
                shift,
                _prior_log_normal.log_normal(mu, 1.0),
            ),
        )

        rtol = 1e-2
        zlo, zhi = model.integrate(rtol=rtol)
        assert zhi - zlo <= rtol * zlo

        # quad comes with an absolute error estimate; be generous with it
        chk, chk_err = scipy.integrate.quad(
            model.mass, 0, 1, epsabs=0, epsrel=1e-8
        )
        assert zlo <= chk + chk_err
        assert zhi >= chk - chk_err
