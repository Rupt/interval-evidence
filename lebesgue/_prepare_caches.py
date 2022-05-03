"""
Cache slow-to-compile functions.

The odd layout here enables numba's caching.

"""
import numba

from . import _bayes, _likelihood_poisson, _prior_log_normal, _prior_plus

example_likelihood_poisson = _likelihood_poisson.poisson(0)
example_prior_log_normal = _prior_log_normal.log_normal(0.0, 1.0)
example_prior_plus_log_normal = _prior_plus.plus(0.0, example_prior_log_normal)


# poisson x log_normal
key = (
    example_likelihood_poisson._interval_func,
    example_prior_log_normal._between_func,
)

_poisson_log_normal = _bayes._integrate_func(*key)


@numba.njit(cache=True)
def poisson_log_normal(args, ratio):
    return _poisson_log_normal(args, ratio)


_bayes._integrate_func_cache[key] = poisson_log_normal


# poisson x plus(log_normal)
key = (
    example_likelihood_poisson._interval_func,
    example_prior_plus_log_normal._between_func,
)

_poisson_plus_log_normal = _bayes._integrate_func(*key)


@numba.njit(cache=True)
def poisson_plus_log_normal(args, ratio):
    return _poisson_plus_log_normal(args, ratio)


_bayes._integrate_func_cache[key] = poisson_plus_log_normal
