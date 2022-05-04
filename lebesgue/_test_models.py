import itertools

import numpy
import scipy.integrate

from . import Model
from .likelihood import poisson
from .prior import add, log_normal, normal, trunc


def test_poisson_add_log_normal():
    ns = [0, 3]
    shifts = [0, 2]
    mus = [0, 5]
    sigmas = [0.5, 1.5]

    for n, shift, mu, sigma in itertools.product(ns, shifts, mus, sigmas):
        model = Model(poisson(n), add(shift, log_normal(mu, sigma)))

        rtol = 1e-2
        zlo, zhi = model.integrate(rtol=rtol)
        assert zhi - zlo <= rtol * zlo

        # quad comes with an absolute error estimate; be generous with it
        chk, chk_err = scipy.integrate.quad(
            model.mass, 0, 1, epsabs=0, epsrel=1e-8
        )
        assert zlo <= chk + chk_err
        assert zhi >= chk - chk_err


def test_poisson_add_trunc_normal():
    ns = [3, 4]
    shifts = [0, 0.2]
    mus = [-1, 2]
    sigmas = [0.5, 1.5]

    for n, shift, mu, sigma in itertools.product(ns, shifts, mus, sigmas):
        model = Model(
            poisson(n), add(0.0, trunc(shift, numpy.inf, normal(mu, sigma)))
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
