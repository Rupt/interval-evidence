import itertools

import scipy.integrate

from .canned import poisson_log_normal, poisson_trunc_normal


def test_poisson_log_normal():
    ns = [0, 3]
    mus = [0, 5]
    sigmas = [0.5, 1.5]
    shifts = [0, 2]

    for n, mu, sigma, shift in itertools.product(ns, mus, sigmas, shifts):
        model = poisson_log_normal(n, mu, sigma, shift=shift)

        rtol = 1e-2
        zlo, zhi = model.integrate(rtol=rtol)
        assert zhi - zlo <= rtol * zlo

        # quad comes with an absolute error estimate; be generous with it
        chk, chk_err = scipy.integrate.quad(
            model.mass, 0, 1, epsabs=0, epsrel=1e-8
        )
        assert zlo <= chk + chk_err
        assert zhi >= chk - chk_err


def test_poisson_trunc_normal():
    ns = [3, 4]
    mus = [-1, 2]
    sigmas = [0.5, 1.5]
    shifts = [0, 0.2]

    for n, mu, sigma, shift in itertools.product(ns, mus, sigmas, shifts):
        lo = mu - 3 * sigma
        hi = mu + 3 * sigma
        model = poisson_trunc_normal(n, mu, sigma, shift=shift, lo=lo, hi=hi)

        rtol = 1e-2
        zlo, zhi = model.integrate(rtol=rtol)
        assert zhi - zlo <= rtol * zlo

        # quad comes with an absolute error estimate; be generous with it
        chk, chk_err = scipy.integrate.quad(
            model.mass, 0, 1, epsabs=0, epsrel=1e-8
        )
        assert zlo <= chk + chk_err
        assert zhi >= chk - chk_err
