import jax
import numpy
import scipy

from .normal_poisson1_transform import p1_of_x_value_and_grad


def test_density():

    normal = scipy.stats.norm().pdf
    poisson1 = scipy.stats.gamma(2).pdf

    x = numpy.linspace(-6, 6, 1000)
    t, dtdx = p1_of_x_value_and_grad(x)
    t = numpy.array(t)
    dtdx = numpy.array(dtdx)

    print(t[:3])
    print(dtdx[:3])
    numpy.testing.assert_allclose(
        normal(x),
        poisson1(t) * dtdx,
        rtol=1e-3,
    )
