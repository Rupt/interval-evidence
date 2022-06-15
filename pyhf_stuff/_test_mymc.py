import jax
import numpy

from .mymc_pyhf import clip_to_x_bounds


def test_clip_to_x_bounds():
    ndim = 3

    def x_of_t(t):
        return 2 * t + 1

    def t_of_x(x):
        return 0.5 * (x - 1)

    def init_func():
        def init(_):
            return jax.numpy.ones(ndim)

        return init

    bounds = jax.numpy.array([[-1, 1], [2, 4], [-3, -1]])

    clip = clip_to_x_bounds(init_func, x_of_t, bounds, t_of_x)

    t_ref = t_of_x(numpy.array([1.0, 3.0, -1.0]))
    t_check = clip()(None)

    numpy.testing.assert_array_equal(t_ref, t_check)
