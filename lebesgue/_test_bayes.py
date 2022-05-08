import numpy

from ._bayes import Likelihood, Model, Prior, _model_mass
from ._test import raises
from .likelihood import poisson
from .prior import log_normal


def test_args_likelihood():
    likelihood = Likelihood(None, lambda args, ratio: (0.0, 0.0))

    assert raises(lambda: likelihood.interval(None), TypeError)
    assert raises(lambda: likelihood.interval(1.1), ValueError)
    assert raises(lambda: likelihood.interval(-0.1), ValueError)
    assert not raises(lambda: likelihood.interval(1.0))
    assert not raises(lambda: likelihood.interval(0.0))
    assert not raises(lambda: likelihood.interval(0.3))


def test_args_prior():
    prior = Prior(None, lambda args, lo, hi: 0.0)

    assert raises(lambda: prior.between(None, 0.0), TypeError)
    assert raises(lambda: prior.between(0.0, -1.0), ValueError)
    assert not raises(lambda: prior.between(1.0, 1.0))
    assert not raises(lambda: prior.between(1.0, 2.0))


def test_args_model():
    likelihood = poisson(0)
    prior = log_normal(0, 1)
    model = Model(likelihood, prior)

    assert not raises(lambda: Model(likelihood, prior))

    assert raises(lambda: model.mass(None), TypeError)
    assert raises(lambda: model.mass(1.1), ValueError)
    assert raises(lambda: model.mass(-0.1), ValueError)

    assert raises(lambda: model.integrate(rtol=None), TypeError)
    assert raises(lambda: model.integrate(rtol=0), ValueError)
    assert not raises(lambda: model.integrate())


def test_monotonic():
    rng = numpy.random.Generator(numpy.random.Philox(10))

    xs = numpy.linspace(0, 1, 20)

    for _ in range(3):
        mu = rng.standard_cauchy()
        sigma = rng.exponential() + 1e-3
        n = rng.geometric(0.1)

        model = Model(poisson(n), log_normal(mu, sigma))

        mlast = model.mass(xs[0])
        for xi in xs[1:]:
            mi = model.mass(xi)
            assert mlast >= mi, (mlast, xi, mi)
            mlast = mi


def test_model_mass():
    likelihood = poisson(0)
    prior = log_normal(0, 1)

    mass = _model_mass(likelihood.interval_func, prior.between_func)

    # caching works
    assert mass is _model_mass(likelihood.interval_func, prior.between_func)

    model = Model(likelihood, prior)

    args = (likelihood.args, prior.args)

    for x in numpy.linspace(0, 1, 14):
        assert model.mass(x) == mass(args, x)
