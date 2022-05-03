import numba
import numpy
import scipy.special

from . import _cephes_ndtr


def test_ndtr():
    rng = numpy.random.Generator(numpy.random.Philox(7))

    assert scipy.special.ndtr(-38) == 0
    assert scipy.special.ndtr(9) == 1

    x = rng.uniform(-38, 9, size=100)

    for x, y_ref in zip(x, scipy.special.ndtr(x)):
        assert _cephes_ndtr.ndtr(x) == y_ref


def test_signatures():
    """Verify that only float arguments are compiled for."""
    # call with integers
    _cephes_ndtr.ndtr(0)

    # check only f8 signatures are compiled
    assert _cephes_ndtr.ndtr.signatures == [(numba.f8,)]
