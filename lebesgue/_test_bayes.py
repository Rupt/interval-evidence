import numpy

from . import _bayes, _likelihood_poisson, _prior_log_normal, _testing


def test_args_likelihood():
    assert _testing.raises(lambda: _bayes._Likelihood(0, None), TypeError)

    likelihood = _bayes._Likelihood(None, lambda args, ratio: (0.0, 0.0))

    assert _testing.raises(lambda: likelihood.interval(None), TypeError)
    assert _testing.raises(lambda: likelihood.interval(1.1), ValueError)
    assert _testing.raises(lambda: likelihood.interval(-0.1), ValueError)
    assert not _testing.raises(lambda: likelihood.interval(1.0))
    assert not _testing.raises(lambda: likelihood.interval(0.0))
    assert not _testing.raises(lambda: likelihood.interval(0.3))


def test_args_prior():
    assert _testing.raises(lambda: _bayes._Prior(0, None), TypeError)

    prior = _bayes._Prior(None, lambda args, lo, hi: 0.0)

    assert _testing.raises(lambda: prior.between(None, 0.0), TypeError)
    assert _testing.raises(lambda: prior.between(0.0, -1.0), ValueError)
    assert not _testing.raises(lambda: prior.between(1.0, 1.0))
    assert not _testing.raises(lambda: prior.between(1.0, 2.0))


def test_args_model():
    likelihood = _likelihood_poisson.poisson(0)
    prior = _prior_log_normal.log_normal(0, 1)

    assert _testing.raises(lambda: _bayes.Model(None, None), TypeError)
    assert _testing.raises(lambda: _bayes.Model(None, prior), TypeError)
    assert _testing.raises(lambda: _bayes.Model(likelihood, None), TypeError)
    assert _testing.raises(lambda: _bayes.Model(prior, likelihood), TypeError)
    assert not _testing.raises(lambda: _bayes.Model(likelihood, prior))


def test_monotonic():
    rng = numpy.random.Generator(numpy.random.Philox(10))

    xs = numpy.linspace(0, 1, 20)

    for _ in range(3):
        mu = rng.standard_cauchy()
        sigma = rng.exponential() + 1e-3
        n = rng.geometric(0.1)

        model = _bayes.Model(
            _likelihood_poisson.poisson(n),
            _prior_log_normal.log_normal(mu, sigma),
        )

        mlast = model.mass(xs[0])
        for xi in xs[1:]:
            mi = model.mass(xi)
            assert mlast >= mi, (mlast, xi, mi)
            mlast = mi


def test_model_mass():
    likelihood = _likelihood_poisson.poisson(0)
    prior = _prior_log_normal.log_normal(0, 1)

    mass = _bayes._model_mass(likelihood.interval_func, prior.between_func)

    # caching works
    assert mass is _bayes._model_mass(
        likelihood.interval_func, prior.between_func
    )

    model = _bayes.Model(likelihood, prior)

    args = (likelihood.args, prior.args)

    for x in numpy.linspace(0, 1, 14):
        assert model.mass(x) == mass(args, x)
