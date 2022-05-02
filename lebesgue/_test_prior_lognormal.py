import itertools

import numpy
import scipy.special

from . import _prior_log_normal, _testing


def test_gaussian_dcdf():
    # gaussian_dcdf is implemented to be better than the reference
    # which gets bad truncation error for large hi, lo
    rng = numpy.random.Generator(numpy.random.Philox(2))

    xys = rng.normal(scale=3, size=(1000, 2))

    for xi, yi in xys:
        chk = _prior_log_normal.gaussian_dcdf(xi, yi)
        ref = gaussian_dcdf_ref(xi, yi)
        numpy.testing.assert_allclose(chk, ref, atol=1e-15)


def test_between():
    rng = numpy.random.Generator(numpy.random.Philox(3))

    ntest = 100
    mus = rng.normal(size=ntest)
    sigmas = rng.exponential(size=ntest) + 1e-15

    for mu, sigma in zip(mus, sigmas):
        prior = _prior_log_normal.log_normal(mu, sigma)

        x, y = sorted(rng.normal(size=2))

        chk = prior.between(numpy.exp(x), numpy.exp(y))
        ref = gaussian_dcdf_ref((x - mu) / sigma, (y - mu) / sigma)

        numpy.testing.assert_allclose(chk, ref, atol=1e-15)


def test_args():
    assert _testing.raises(lambda: _prior_log_normal.log_normal(None, 1))
    assert _testing.raises(lambda: _prior_log_normal.log_normal(1, None))
    assert _testing.raises(lambda: _prior_log_normal.log_normal(1, -1))


# utilities


def gaussian_dcdf_ref(lo, hi):
    return scipy.special.ndtr(hi) - scipy.special.ndtr(lo)
