"""
Cache slow-to-compile functions.

The odd layout here enables numba's caching.

"""
import numba

from . import _quad_bound
from ._bayes import Model
from ._likelihood_poisson import poisson
from ._prior_log_normal import log_normal
from ._prior_plus import plus

# poisson | log normal
model = Model(poisson(0), log_normal(0, 1))

poisson_log_normal_integrate = model.integrate_func


@numba.njit(cache=True)
def poisson_log_normal(args, ratio):
    return poisson_log_normal_integrate(args, ratio)


_quad_bound._generate_cache[model.mass_func] = poisson_log_normal


# poisson | plus log normal
model = Model(poisson(0), plus(1, log_normal(0, 1)))

poisson_plus_log_normal_integrate = model.integrate_func


@numba.njit(cache=True)
def poisson_plus_log_normal(args, ratio):
    return poisson_plus_log_normal_integrate(args, ratio)


_quad_bound._generate_cache[model.mass_func] = poisson_plus_log_normal
