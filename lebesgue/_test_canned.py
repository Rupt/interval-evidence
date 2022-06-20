import scipy.integrate

from .canned import (
    gamma1_log_normal,
    gamma1_regular_linear,
    gamma1_regular_uniform,
    gamma1_trunc_normal,
    poisson_log_normal,
    poisson_regular_linear,
    poisson_regular_uniform,
    poisson_trunc_normal,
)


def test_poisson_log_normal():
    n_mu_sigma_shifts = [
        (0, 0, 0.5, 0),
        (3, 1, 1.5, 2),
    ]

    for n, mu, sigma, shift in n_mu_sigma_shifts:
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
    n_mu_sigma_shift_lo_his = [
        (0, -1, 0.5, 0, 0, 1),
        (3, 2, 1.5, 0.2, 1, 2.5),
    ]

    for n, mu, sigma, shift, lo, hi in n_mu_sigma_shift_lo_his:
        model = poisson_trunc_normal(n, mu, sigma, shift=shift, lo=lo, hi=hi)

        rtol = 1e-2
        zlo, zhi = model.integrate(rtol=rtol)
        assert zhi - zlo <= rtol * zlo, (zlo, zhi)

        # quad comes with an absolute error estimate; be generous with it
        chk, chk_err = scipy.integrate.quad(
            model.mass, 0, 1, epsabs=0, epsrel=1e-8
        )
        assert zlo <= chk + chk_err
        assert zhi >= chk - chk_err


def test_poisson_regular_linear():
    n = 3
    start = 0
    stop = 5
    log_rates = [-10, -1, 0, 0, 1, 0, -1]
    shift = 0.5

    model = poisson_regular_linear(n, start, stop, log_rates, shift=shift)

    rtol = 1e-2
    zlo, zhi = model.integrate(rtol=rtol)
    assert zhi - zlo <= rtol * zlo

    # quad comes with an absolute error estimate; be generous with it
    chk, chk_err = scipy.integrate.quad(
        model.mass, 0, 1, epsabs=0, epsrel=1e-8
    )
    assert zlo <= chk + chk_err
    assert zhi >= chk - chk_err


def test_poisson_regular_uniform():
    n = 10
    start = 0
    stop = 5
    log_rates = [1, 2, 3, 2, 1]
    shift = 4.2

    model = poisson_regular_uniform(n, start, stop, log_rates, shift=shift)

    rtol = 1e-2
    zlo, zhi = model.integrate(rtol=rtol)
    assert zhi - zlo <= rtol * zlo

    # quad comes with an absolute error estimate; be generous with it
    chk, chk_err = scipy.integrate.quad(
        model.mass, 0, 1, epsabs=0, epsrel=1e-8
    )
    assert zlo <= chk + chk_err
    assert zhi >= chk - chk_err


def test_same_funcs():
    assert (
        poisson_log_normal(0, 1, 2).integrate_func
        is gamma1_log_normal(0, 1, 2).integrate_func
    )
    assert (
        gamma1_trunc_normal(0, 1, 2).integrate_func
        is gamma1_trunc_normal(0, 1, 2).integrate_func
    )
    assert (
        poisson_regular_linear(0, 0, 1, [0, 1]).integrate_func
        is gamma1_regular_linear(0, 0, 1, [0, 1]).integrate_func
    )
    assert (
        poisson_regular_uniform(0, 0, 1, [1]).integrate_func
        is gamma1_regular_uniform(0, 0, 1, [1]).integrate_func
    )
