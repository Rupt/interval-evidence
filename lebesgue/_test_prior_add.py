import numpy

from . import _testing
from .likelihood import poisson
from .prior import add, log_normal


def test_shift():
    prior = log_normal(1, 3)

    rng = numpy.random.Generator(numpy.random.Philox(4))

    for x, lo, delta in rng.normal(size=(100, 3)):
        hi = lo + abs(delta)

        # such that lo + x == lo_previous
        ref = prior.between(lo - x, hi - x)
        chk = add(x, prior).between(lo, hi)

        assert ref == chk, (ref, chk)


def test_args():
    prior = log_normal(-4, 2.0)
    assert _testing.raises(lambda: add(None, prior), TypeError)

    not_prior = poisson(3)
    assert _testing.raises(lambda: add(3, not_prior), AttributeError)

    assert not _testing.raises(lambda: add(0.0, prior))
