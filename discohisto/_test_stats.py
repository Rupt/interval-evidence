import numpy
import scipy.stats

from .stats import llr_to_sigma, poisson_log_minus_max, sigma_to_llr


def test_sigma_to_fro_llr():
    # check the standard examples
    sigma_llr_ref = [
        (0, 0),
        (1, 0.5),
        (2, 2),
        (3, 4.5),
        (4, 8),
        (5, 12.5),
        (6, 18),
    ]

    for sigma, llr in sigma_llr_ref:
        assert sigma_to_llr(sigma) == llr
        assert sigma_to_llr(-sigma) == -llr
        assert llr_to_sigma(llr) == sigma
        assert llr_to_sigma(-llr) == -sigma

    # check on numpy array data
    rng = numpy.random.Generator(numpy.random.Philox(1))

    llr = rng.standard_cauchy(size=1000) * 2

    sigma = llr_to_sigma(llr)
    llr_check = sigma_to_llr(sigma)

    numpy.testing.assert_allclose(llr, llr_check)

    # check sequences work
    numpy.testing.assert_array_equal(
        sigma_to_llr(range(10)), sigma_to_llr(numpy.arange(10))
    )

    numpy.testing.assert_array_equal(
        llr_to_sigma(range(10)), llr_to_sigma(numpy.arange(10))
    )


def test_poisson_log_minus_max():
    rng = numpy.random.Generator(numpy.random.Philox(123))

    # fuzz
    n = rng.integers(1000, size=2000)
    mu = rng.uniform(high=1000, size=len(n))

    chk = poisson_log_minus_max(n, mu)
    ref = scipy.stats.poisson.logpmf(n, mu) - scipy.stats.poisson.logpmf(n, n)
    numpy.testing.assert_allclose(chk, ref)

    # special cases
    n, mu = numpy.array([
        (0, 1.0),
        (0, 0.0),
        (1, 0.0),
    ]).T
    chk = poisson_log_minus_max(n, mu)
    ref = scipy.stats.poisson.logpmf(n, mu) - scipy.stats.poisson.logpmf(n, n)
    numpy.testing.assert_allclose(chk, ref)
