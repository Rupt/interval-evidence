""" Canned laughter haha xaxa jaja. But for models. """
import numba
import numpy

from ._bayes import Model
from ._quad_bound import _integrator_cache
from .likelihood import poisson
from .prior import add, log_normal, normal, trunc


def poisson_log_normal(
    n: int, mu: float, sigma: float, *, shift: float = 0.0
) -> Model:
    """Return a Model(poisson(n) | log_normal(mu, sigma) + shift)."""
    return Model(poisson(n), add(shift, log_normal(mu, sigma)))


def poisson_trunc_normal(
    n: int,
    mu: float,
    sigma: float,
    *,
    shift: float = 0.0,
    lo: float = 0.0,
    hi: float = numpy.inf
) -> Model:
    """Return a Model(poisson(n) | log_normal(mu, sigma) + shift)."""
    return Model(poisson(n), add(shift, trunc(lo, hi, normal(mu, sigma))))


# prepare caches to reduce expensive recompilations
# this quirky layout allows numba to cache successfully


def integrate_signature(args):
    pair_float64 = numba.typeof((0.0, 0.0))
    typeof_args = numba.typeof(args)
    return pair_float64(typeof_args, numba.float64)


# poisson_log_normal
_model = poisson_log_normal(0, 0, 1)

_poisson_log_normal_integrate_func = _model.integrate_func


@numba.njit(integrate_signature(_model.args), cache=True)
def _poisson_log_normal_integrate(args, ratio):
    return _poisson_log_normal_integrate_func(args, ratio)


_integrator_cache[_model.mass_func] = _poisson_log_normal_integrate


# poisson_trunc_normal

_model = poisson_trunc_normal(0, 0, 1)

_poisson_trunc_normal_integrate_func = _model.integrate_func


@numba.njit(integrate_signature(_model.args), cache=True)
def _poisson_trunc_normal_integrate(args, ratio):
    return _poisson_trunc_normal_integrate_func(args, ratio)


_integrator_cache[_model.mass_func] = _poisson_trunc_normal_integrate
