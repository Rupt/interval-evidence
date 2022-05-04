"""
Cache slow-to-compile functions.

The odd layout here enables numba's caching.

"""
import numba
import numpy

from . import _quad_bound
from ._bayes import Model
from .likelihood import poisson
from .prior import log_normal, normal, plus, trunc

# poisson | plus log normal
model = Model(poisson(0), plus(1, log_normal(0, 1)))

poisson_plus_log_normal_integrate = model.integrate_func


@numba.njit(cache=True)
def poisson_plus_log_normal(args, ratio):
    return poisson_plus_log_normal_integrate(args, ratio)


_quad_bound._integrator_cache[model.mass_func] = poisson_plus_log_normal


# poisson | plus trunc normal
model = Model(poisson(0), plus(1, trunc(0, numpy.inf, normal(0, 1))))

poisson_plus_trunc_normal_integrate = model.integrate_func


@numba.njit(cache=True)
def poisson_plus_trunc_normal(args, ratio):
    return poisson_plus_trunc_normal_integrate(args, ratio)


_quad_bound._integrator_cache[model.mass_func] = poisson_plus_trunc_normal
