"""Manage likelihoods and priors."""
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba

from . import _quad_bound


@dataclass(frozen=True)
class _Likelihood:
    _args: Any
    _interval_func: Callable

    def __post_init__(self):
        if not isinstance(self._interval_func, Callable):
            raise TypeError(self._interval_func)

    def interval(self, ratio: float) -> (float, float):
        """Return (lo, hi) representing an interval of likelihood above ratio.

        The returned interval should contain all parameter space for which the
        represented likelihood function exceeds ratio times its maximum.

        Arguments:
            ratio: in [0, 1]
        """
        ratio = float(ratio)
        if not 0 <= ratio <= 1:
            raise ValueError(ratio)
        return self._interval_func(self._args, ratio)


@dataclass(frozen=True)
class _Prior:
    _args: Any
    _between_func: Callable

    def __post_init__(self):
        if not isinstance(self._between_func, Callable):
            raise TypeError(self._between_func)

    def between(self, lo: float, hi: float) -> float:
        """Return the probability mass between lo and hi.

        Arguments:
            lo: low edge
            hi: high edge
        """

        lo = float(lo)
        hi = float(hi)
        if not lo <= hi:
            raise ValueError((lo, hi))
        return self._between_func(self._args, lo, hi)


@dataclass(frozen=True)
class Model:
    """A Likelihood--Prior pair.

    Arguments:
        likelihood: observed data component; gives intervals above ratios
        prior: distribution of probability; gives proportions between limits

    """

    likelihood: _Likelihood
    prior: _Prior

    def __post_init__(self):
        if not isinstance(self.likelihood, _Likelihood):
            raise TypeError(self.likelihood)

        if not isinstance(self.prior, _Prior):
            raise TypeError(self.prior)

    def integrate(self, *, rtol: float = 1e-2) -> (float, float):
        """Return numerical bounds on the integral of likelihood over prior.

        Arguments:
            rtol: float, relative tolerance of the bounds.
                Should satisfy lo < integral < hi
                and (hi - lo) < rtol * lo

                Time cost scales something like ~ 1 / rtol.
        """
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

    def mass(self, ratio: float) -> float:
        """Return the prior mass inside the interval at likelihood ratio."""
        lo, hi = self.likelihood.interval(ratio)
        return self.prior.between(lo, hi)


# caching reduces recompilation, which is expensive
_integrate_func_cache = {}


def _integrate_func(interval_func, between_func):
    key = (interval_func, between_func)
    cached = _integrate_func_cache.get(key)

    if cached is not None:
        return cached

    @numba.njit
    def _mass(args, ratio):
        interval_args, between_args = args
        lo, hi = interval_func(interval_args, ratio)
        return between_func(between_args, lo, hi)

    integrate_func = _quad_bound.generate(_mass)
    _integrate_func_cache[key] = integrate_func
    return integrate_func
