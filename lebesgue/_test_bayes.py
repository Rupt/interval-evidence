import numpy

from . import _bayes, _testing
from .likelihood import poisson
from .prior import log_normal


def test_args_likelihood():
    likelihood = _bayes._Likelihood(None, lambda args, ratio: (0.0, 0.0))

    assert _testing.raises(lambda: likelihood.interval(None), TypeError)
    assert _testing.raises(lambda: likelihood.interval(1.1), ValueError)
    assert _testing.raises(lambda: likelihood.interval(-0.1), ValueError)
    assert not _testing.raises(lambda: likelihood.interval(1.0))
    assert not _testing.raises(lambda: likelihood.interval(0.0))
    assert not _testing.raises(lambda: likelihood.interval(0.3))


def test_args_prior():
    prior = _bayes._Prior(None, lambda args, lo, hi: 0.0)

    assert _testing.raises(lambda: prior.between(None, 0.0), TypeError)
    assert _testing.raises(lambda: prior.between(0.0, -1.0), ValueError)
    assert not _testing.raises(lambda: prior.between(1.0, 1.0))
    assert not _testing.raises(lambda: prior.between(1.0, 2.0))


def test_args_model():
    likelihood = poisson(0)
    prior = log_normal(0, 1)
    model = _bayes.Model(likelihood, prior)

    assert not _testing.raises(lambda: _bayes.Model(likelihood, prior))

    assert _testing.raises(lambda: model.mass(None), TypeError)
    assert _testing.raises(lambda: model.mass(1.1), ValueError)
    assert _testing.raises(lambda: model.mass(-0.1), ValueError)

    assert _testing.raises(lambda: model.integrate(rtol=None), TypeError)
    assert _testing.raises(lambda: model.integrate(rtol=0), ValueError)
    assert not _testing.raises(lambda: model.integrate())


def test_monotonic():
    rng = numpy.random.Generator(numpy.random.Philox(10))

    xs = numpy.linspace(0, 1, 20)

    for _ in range(3):
        mu = rng.standard_cauchy()
        sigma = rng.exponential() + 1e-3
        n = rng.geometric(0.1)

        model = _bayes.Model(poisson(n), log_normal(mu, sigma))

        mlast = model.mass(xs[0])
        for xi in xs[1:]:
            mi = model.mass(xi)
            assert mlast >= mi, (mlast, xi, mi)
            mlast = mi


def test_model_mass():
    likelihood = poisson(0)
    prior = log_normal(0, 1)

    mass = _bayes._model_mass(likelihood.interval_func, prior.between_func)

    # caching works
    assert mass is _bayes._model_mass(
        likelihood.interval_func, prior.between_func
    )

    model = _bayes.Model(likelihood, prior)

    args = (likelihood.args, prior.args)

    for x in numpy.linspace(0, 1, 14):
        assert model.mass(x) == mass(args, x)
