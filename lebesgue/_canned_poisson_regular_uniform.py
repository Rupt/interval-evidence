import numba

from ._bayes import Model
from ._quad_bound import integrator, integrator_signature
from .likelihood import gamma1, poisson
from .prior import add, regular_uniform


def poisson_regular_uniform(
    n: int,
    start: float,
    stop: float,
    log_rates: list[float],
    *,
    shift: float = 0.0
) -> Model:
    """Return a Model(poisson(n) | regular_uniform + shift)."""
    return Model(
        poisson(n), add(shift, regular_uniform(start, stop, log_rates))
    )


def gamma1_regular_uniform(
    shape: float,
    start: float,
    stop: float,
    log_rates: list[float],
    *,
    shift: float = 0.0
) -> Model:
    """Return a Model(gamma1(shape) | regular_uniform + shift)."""
    return Model(
        gamma1(shape), add(shift, regular_uniform(start, stop, log_rates))
    )


# integrator caching


example_model = poisson_regular_uniform(0, 0, 1, [1, 2])

integrate_func = example_model.integrate_func


@integrator.put(example_model.mass_func)
@numba.njit(integrator_signature(example_model.args), cache=True)
def integrate(args, rtol):
    return integrate_func(args, rtol)
