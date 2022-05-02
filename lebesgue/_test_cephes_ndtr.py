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


def test_erf():
    rng = numpy.random.Generator(numpy.random.Philox(8))

    assert scipy.special.erf(-6) == -1
    assert scipy.special.erf(6) == 1

    x = rng.uniform(-6, 6, size=100)

    for x, y_ref in zip(x, scipy.special.erf(x)):
        assert _cephes_ndtr.erf(x) == y_ref


def test_erfc():
    rng = numpy.random.Generator(numpy.random.Philox(9))

    assert scipy.special.erfc(-6) == 2
    assert scipy.special.erfc(27) == 0

    x = rng.uniform(-38, 9, size=100)

    for x, y_ref in zip(x, scipy.special.erfc(x)):
        assert _cephes_ndtr.erfc(x) == y_ref


def test_signatures():
    """Verify that only float arguments are compiled for."""
    # call with integers
    _cephes_ndtr.ndtr(0)
    _cephes_ndtr.erf(0)
    _cephes_ndtr.erfc(0)

    # check only f8 signatures are compiled
    assert _cephes_ndtr.ndtr.signatures == [(numba.f8,)]
    assert _cephes_ndtr.erf.signatures == [(numba.f8,)]
    assert _cephes_ndtr.erfc.signatures == [(numba.f8,)]
