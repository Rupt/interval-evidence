"""
Cache slow-to-compile functions.

The odd layout here enables numba's caching.

"""
import numba

# TODO fix black isort fight here
from . import (
    _bayes,
    _likelihood_poisson,
    _prior_log_normal,
    _prior_plus,
    _quad_bound,
)

example_likelihood_poisson = _likelihood_poisson.poisson(0)
example_prior_log_normal = _prior_log_normal.log_normal(0.0, 1.0)
example_prior_plus_log_normal = _prior_plus.plus(0.0, example_prior_log_normal)


# poisson x log_normal
mass_func = _bayes._model_mass(
    example_likelihood_poisson.interval_func,
    example_prior_log_normal.between_func,
)

_poisson_log_normal = _quad_bound.generate(mass_func)


@numba.njit(cache=True)
def poisson_log_normal(args, ratio):
    return _poisson_log_normal(args, ratio)


_quad_bound._generate_cache[mass_func] = poisson_log_normal


# poisson x plus(log_normal)
mass_func = _bayes._model_mass(
    example_likelihood_poisson.interval_func,
    example_prior_plus_log_normal.between_func,
)

_poisson_plus_log_normal = _quad_bound.generate(mass_func)


@numba.njit(cache=True)
def poisson_plus_log_normal(args, ratio):
    return _poisson_plus_log_normal(args, ratio)


_quad_bound._generate_cache[mass_func] = poisson_plus_log_normal
