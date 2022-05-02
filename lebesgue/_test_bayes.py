from . import _bayes, _likelihood_poisson, _prior_log_normal, _testing


def test_args_likelihood():
    assert _testing.raises(lambda: _bayes.Likelihood(None), TypeError)
    assert _testing.raises(
        lambda: _bayes.Likelihood(_prior_log_normal._LogNormal(0, 1)), TypeError
    )
    assert not _testing.raises(
        lambda: _bayes.Likelihood(_likelihood_poisson._Poisson(0))
    )


def test_args_prior():
    assert _testing.raises(lambda: _bayes.Prior(None), TypeError)
    assert _testing.raises(
        lambda: _bayes.Prior(_likelihood_poisson._Poisson(0)), TypeError
    )
    assert not _testing.raises(lambda: _bayes.Prior(_prior_log_normal._LogNormal(0, 1)))


def test_args_model():
    prior = _prior_log_normal.log_normal(0, 1)
    likelihood = _likelihood_poisson.poisson(0)

    assert _testing.raises(lambda: _bayes.Model(None, None), TypeError)
    assert _testing.raises(lambda: _bayes.Model(None, prior), TypeError)
    assert _testing.raises(lambda: _bayes.Model(likelihood, None), TypeError)
    assert _testing.raises(lambda: _bayes.Model(prior, likelihood), TypeError)
    assert not _testing.raises(lambda: _bayes.Model(likelihood, prior))
