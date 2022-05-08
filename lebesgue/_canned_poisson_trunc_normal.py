import numba

from ._bayes import Model
from ._quad_bound import integrator, integrator_signature
from .likelihood import poisson
from .prior import add, normal, trunc


def poisson_trunc_normal(
    n: int,
    mu: float,
    sigma: float,
    *,
    shift: float = 0.0,
    lo: float = 0.0,
    hi: float = float("inf")
) -> Model:
    """Return a Model(poisson(n) | log_normal(mu, sigma) + shift)."""
    return Model(poisson(n), add(shift, trunc(lo, hi, normal(mu, sigma))))


example_model = poisson_trunc_normal(0, 0, 1)

poisson_trunc_normal_integrate_func = example_model.integrate_func


@integrator.put(example_model.mass_func)
@numba.njit(integrator_signature(example_model.args), cache=True)
def poisson_trunc_normal_integrate(args, ratio):
    return poisson_trunc_normal_integrate_func(args, ratio)
