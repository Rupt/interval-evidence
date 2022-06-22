"""Transform a "poisson(1)" constraint "density" to normal coordinates."""
import jax

_P1_OF_X_LO = -4.5
_P1_OF_X_HI = 4.5



def p1_of_x(x):
    lo = _p1_of_x_lo(x)
    hi = _p1_of_x_hi(x)
    mid = _p1_of_x_mid(x)
    return jax.numpy.where(
        x < _P1_OF_X_LO,
        lo,
        jax.numpy.where(x > _P1_OF_X_HI, hi, mid),
    )

p1_of_x_value_and_grad = jax.jit(jax.vmap(jax.value_and_grad(p1_of_x)))


def _p1_of_x_lo(x):
    s = jax.scipy.special.ndtr(x)
    return (2 * s) ** 0.5 + (2 / 3) * s


def _p1_of_x_hi(x):
    # using an improvement on
    # y = ndtr(x)
    # s = (y - 1) / numpy.e
    # expansion of real W_{-1}(s) at s=0
    # a1 = L1 = log(-s), a2 = L2
    a1 = jax.numpy.log(jax.scipy.special.ndtr(-x)) - 1
    a2 = jax.numpy.log(-a1)

    c3 = 1 + a2 * (-(3 / 2) + (1 / 3) * a2)
    c2 = 0.5 * a2 - 1

    o1 = 1 / a1
    r = a2 * o1 * (1 + o1 * (c2 + o1 * c3))
    return -1 - r - a1 + a2


def _p1_of_x_mid(x):
    chebyfit = jax.numpy.array(
        [
            -7.659900359477959e-10,
            -1.0866672728787215e-09,
            7.934368455942802e-08,
            -1.1524837085802057e-07,
            -3.0834115781903265e-06,
            1.735531594751684e-05,
            7.656902204777025e-06,
            -0.0009082465888769052,
            0.010755071695771322,
            -0.09254122774217419,
            0.758653944415632,
            0.5178028572395069,
        ]
    )
    return jax.numpy.exp(jax.numpy.polyval(chebyfit, x))
