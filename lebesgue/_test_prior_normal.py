from test import raises

import numpy
import scipy.special

from ._prior_normal import _gaussian_dcdf
from .prior import log_normal, normal


def test_gaussian_dcdf():
    # gaussian_dcdf is implemented to be better than the reference
    # which gets bad truncation error for large hi, lo
    rng = numpy.random.Generator(numpy.random.Philox(2))

    xys = rng.normal(scale=3, size=(100, 2))

    for xi, yi in xys:
        chk = _gaussian_dcdf(xi, yi)
        ref = gaussian_dcdf_ref(xi, yi)
        numpy.testing.assert_allclose(chk, ref, atol=1e-15)


def test_between():
    rng = numpy.random.Generator(numpy.random.Philox(3))

    ntest = 100
    mus = rng.normal(size=ntest)
    sigmas = rng.exponential(size=ntest) + 1e-15

    for mu, sigma in zip(mus, sigmas):
        x, y = sorted(rng.normal(size=2))

        ref = gaussian_dcdf_ref((x - mu) / sigma, (y - mu) / sigma)

        # normal
        prior = normal(mu, sigma)
        chk = prior.between(x, y)
        numpy.testing.assert_allclose(chk, ref, atol=1e-15)

        # log_normal
        log_prior = log_normal(mu, sigma)
        chk_log = log_prior.between(numpy.exp(x), numpy.exp(y))
        numpy.testing.assert_allclose(chk_log, ref, atol=1e-15)


def test_args():
    assert raises(lambda: normal(None, 1), TypeError)
    assert raises(lambda: normal(1, None), TypeError)
    assert raises(lambda: normal(2, -1), ValueError)
    assert not raises(lambda: normal(3, 0.1))

    assert raises(lambda: log_normal(None, 1), TypeError)
    assert raises(lambda: log_normal(1, None), TypeError)
    assert raises(lambda: log_normal(2, -1), ValueError)
    assert not raises(lambda: log_normal(3, 0.1))


# utilities


def gaussian_dcdf_ref(lo, hi):
    return scipy.special.ndtr(hi) - scipy.special.ndtr(lo)
