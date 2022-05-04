"""Manage likelihoods and priors."""
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba

from . import _quad_bound


@dataclass(frozen=True)
class _Likelihood:
    args: Any
    interval_func: Callable

    def __post_init__(self):
        if not isinstance(self.interval_func, Callable):
            raise TypeError(self.interval_func)

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
        return self.interval_func(self.args, ratio)


@dataclass(frozen=True)
class _Prior:
    args: Any
    between_func: Callable

    def __post_init__(self):
        if not isinstance(self.between_func, Callable):
            raise TypeError(self.between_func)

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
        return self.between_func(self.args, lo, hi)


@dataclass(frozen=True, init=False)
class Model:
    """A Likelihood--Prior pair.

    Arguments:
        likelihood: observed data component; gives intervals above ratios
        prior: distribution of probability; gives proportions between limits

    """

    likelihood: _Likelihood
    prior: _Prior

    mass_func: Callable
    integrate_func: Callable

    def __init__(self, likelihood: _Likelihood, prior: _Prior) -> None:
        if not isinstance(likelihood, _Likelihood):
            raise TypeError(likelihood)

        if not isinstance(prior, _Prior):
            raise TypeError(prior)

        mass_func = _model_mass(likelihood.interval_func, prior.between_func)
        integrate_func = _quad_bound.integrator(mass_func)

        object.__setattr__(self, "likelihood", likelihood)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "mass_func", mass_func)
        object.__setattr__(self, "integrate_func", integrate_func)

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

        args = (self.likelihood.args, self.prior.args)
        return self.integrate_func(args, rtol)

    def mass(self, ratio: float) -> float:
        """Return the prior mass inside the interval at likelihood ratio."""
        lo, hi = self.likelihood.interval(ratio)
        return self.prior.between(lo, hi)


@functools.lru_cache(maxsize=None)
def _model_mass(interval_func, between_func):
    @numba.njit
    def _mass(args, ratio):
        args_interval, args_between = args
        lo, hi = interval_func(args_interval, ratio)
        return between_func(args_between, lo, hi)

    return _mass
