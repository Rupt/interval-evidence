import numpy
import scipy.special

from ._cephes_ndtr import ndtr


def test_ndtr():
    rng = numpy.random.Generator(numpy.random.Philox(7))

    assert scipy.special.ndtr(-38) == 0
    assert scipy.special.ndtr(9) == 1

    x = rng.uniform(-38, 9, size=100)

    for x, y_ref in zip(x, scipy.special.ndtr(x)):
        assert ndtr(x) == y_ref
