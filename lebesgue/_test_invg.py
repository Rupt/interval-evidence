import numpy

from ._invg import _invg_hi, _invg_lo, ginterval


def test_invg_lo():
    xs = numpy.linspace(1e-3, 1, 1000)

    for x_ref in xs:
        y_ref = gfunc(x_ref)

        x_chk = _invg_lo(y_ref)
        y_chk = gfunc(x_chk)

        assert ulp_min(x_ref, y_ref, x_chk, y_chk) <= 2


def test_invg_hi():
    xs = numpy.linspace(1, 1e3, 1000)

    for x_ref in xs:
        y_ref = gfunc(x_ref)

        x_chk = _invg_hi(y_ref)
        y_chk = gfunc(x_chk)

        assert ulp_min(x_ref, y_ref, x_chk, y_chk) <= 1


def test_ginterval():
    assert ginterval(0, 0) == (0.0, numpy.inf)
    assert numpy.isnan(ginterval(0, -1)).all()
    assert ginterval(0, 0.4) == (0.0, -numpy.log(0.4))

    shape_and_ratios = [
        (0.5, 0.1),
        (0.1, 0.1),
        (0.2, 0.5),
        (4.0, 1.0),
    ]

    for shape, ratio in shape_and_ratios:
        lo, hi = ginterval(shape, ratio)

        numpy.testing.assert_allclose(
            -shape * gfunc(lo / shape),
            numpy.log(ratio),
        )


# utilities


def gfunc(x):
    return x - 1 - numpy.log(x)


def ulp_min(x_ref, y_ref, x_chk, y_chk):
    # either close in the function or its inverse
    return min(
        abs(x_ref - x_chk) / numpy.spacing(x_ref),
        abs(y_ref - y_chk) / numpy.spacing(y_ref),
    )
