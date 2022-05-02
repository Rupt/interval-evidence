from . import _likelihood_poisson, _testing


def test_args():
    assert _testing.raises(lambda: _likelihood_poisson.poisson(None), TypeError)
    assert _testing.raises(lambda: _likelihood_poisson.poisson(-1), ValueError)
    assert _testing.raises(lambda: _likelihood_poisson.poisson(0.5), ValueError)
    assert not _testing.raises(lambda: _likelihood_poisson.poisson(0))
    assert not _testing.raises(lambda: _likelihood_poisson.poisson(1))
