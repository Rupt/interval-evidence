import numpy

from . import _testing
from .prior import normal, trunc


def test_values():
    prior = normal(0, 1)

    pairs = [
        (-1e6, 1e6),
        (0, 1e6),
        (0, 1),
        (1, 2),
        (1 - 1e-8, 1),
        (-1.5, -0.5),
    ]

    for lo, hi in pairs:
        trunc_prior = trunc(lo, hi, prior)

        # full interval is normalized
        assert trunc_prior.between(-numpy.inf, numpy.inf) == 1

        # proportion within the bounds is changed by normalization
        lo_inside = lo + 0.25 * (hi - lo)
        hi_inside = lo + 0.75 * (hi - lo)

        norm = prior.between(lo, hi)

        ref = prior.between(lo_inside, hi_inside) / norm
        chk = trunc_prior.between(lo_inside, hi_inside)
        assert ref == chk, (ref, chk)

    # a case we previously failed with negative between
    trunc_prior = trunc(-2.5, -0.5, normal(-1, 0.5))
    chk = trunc_prior.between(1, 5)
    assert chk >= 0, chk


def test_args():
    prior = normal(0, 1)

    assert _testing.raises(lambda: trunc(1, 0, prior), ValueError)
    assert _testing.raises(lambda: trunc(1, None, prior), TypeError)
    assert _testing.raises(lambda: trunc(None, 1, prior), TypeError)
