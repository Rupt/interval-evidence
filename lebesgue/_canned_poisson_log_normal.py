import numba

from ._bayes import Model
from ._quad_bound import integrator, integrator_signature
from .likelihood import gamma1, poisson
from .prior import add, log_normal


def poisson_log_normal(
    n: int, mu: float, sigma: float, *, shift: float = 0.0
) -> Model:
    """Return a Model(poisson(n) | log_normal(mu, sigma) + shift)."""
    return Model(poisson(n), add(shift, log_normal(mu, sigma)))


def gamma1_log_normal(
    shape: float, mu: float, sigma: float, *, shift: float = 0.0
) -> Model:
    """Return a Model(gamma1(shape) | log_normal(mu, sigma) + shift)."""
    return Model(gamma1(shape), add(shift, log_normal(mu, sigma)))


# integrator caching


example_model = poisson_log_normal(0, 0, 1)

integrate_func = example_model.integrate_func


@integrator.put(example_model.mass_func)
@numba.njit(integrator_signature(example_model.args), cache=True)
def integrate(args, ratio):
    return integrate_func(args, ratio)
