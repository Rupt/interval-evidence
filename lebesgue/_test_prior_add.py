import numpy

from . import _likelihood_poisson, _prior_add, _prior_normal, _testing


def test_shift():
    prior = _prior_normal.log_normal(1, 3)

    rng = numpy.random.Generator(numpy.random.Philox(4))

    for x, lo, delta in rng.normal(size=(100, 3)):
        hi = lo + abs(delta)

        # such that lo + x == lo_previous
        ref = prior.between(lo - x, hi - x)
        chk = _prior_add.add(x, prior).between(lo, hi)

        assert ref == chk, (ref, chk)


def test_args():
    prior = _prior_normal.log_normal(-4, 2.0)
    assert _testing.raises(lambda: _prior_add.add(None, prior), TypeError)

    not_prior = _likelihood_poisson.poisson(3)
    assert _testing.raises(lambda: _prior_add.add(3, not_prior), TypeError)

    assert not _testing.raises(lambda: _prior_add.add(0.0, prior))
