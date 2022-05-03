"""Manage likelihoods and priors."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba
from numba.core.types.misc import ClassInstanceType

from . import _core, _quad_bound

# python-facing


@dataclass(frozen=True)
class Likelihood:
    _args: Any
    _interval_func: Callable

    def __post_init__(self):
        if not isinstance(self._interval_func, Callable):
            raise TypeError(self._interval_func)

    def interval(self, ratio):
        ratio = float(ratio)
        if not 0 <= ratio <= 1:
            raise ValueError(ratio)
        return self._interval_func(self._args, ratio)


@dataclass(frozen=True)
class Prior:
    _args: Any
    _between_func: Callable

    def __post_init__(self):
        if not isinstance(self._between_func, Callable):
            raise TypeError(self._between_func)

    def between(self, lo, hi):
        lo = float(lo)
        hi = float(hi)
        if not lo <= hi:
            raise ValueError((lo, hi))
        return self._between_func(self._args, lo, hi)


@dataclass(frozen=True)
class Model:
    likelihood: Likelihood
    prior: Prior

    def __post_init__(self):
        if not isinstance(self.likelihood, Likelihood):
            raise TypeError(self.likelihood)

        if not isinstance(self.prior, Prior):
            raise TypeError(self.prior)

    def integrate(self, *, rtol=1e-2):
        rtol = float(rtol)
        # tiny tol (< 2 ** -51?) can cause runaway recursion
        # small tol is slow and unlikely to be useful
        assert rtol >= 1e-7, rtol

        args = (self.likelihood._args, self.prior._args)
        integrate_func = _integrate_func(
            self.likelihood._interval_func,
            self.prior._between_func,
        )

        return integrate_func(args, rtol)

    def mass(self, ratio):
        lo, hi = self.likelihood.interval(ratio)
        return self.prior.between(lo, hi)


# numba-facing
_integrate_func_cache = {}


def _integrate_func(interval_func, between_func):

    key = (interval_func, between_func)

    cached = _integrate_func_cache.get(key)

    if cached is not None:
        return cached

    @_core.jit
    def _mass(args, ratio):
        interval_args, between_args = args
        lo, hi = interval_func(interval_args, ratio)
        return between_func(between_args, lo, hi)

    integrate_func = _quad_bound.generate(_mass)

    _integrate_func_cache[key] = integrate_func

    return integrate_func
