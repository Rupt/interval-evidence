"""Manage likelihoods and priors."""
import functools
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numba

from . import _quad_bound


@dataclass(frozen=True)
class Likelihood:
    """Package a likelihood interval function with arguments.

    Fields:
        args: arguments to pass to interval_func
        interval_func: (args, ratio) -> (lo, hi)
    """

    args: Any
    interval_func: Callable

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
class Prior:
    """Package a prior mass between function with arguments.

    Fields:
        args: arguments to pass to between_func
        interval_func: (args, lo, hi) -> proportion

    """

    args: Any
    between_func: Callable

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


@dataclass(frozen=True)
class Model:
    """A Likelihood--Prior pair.

    Fields:
        likelihood: observed data component; gives intervals above ratios
        prior: distribution of probability; gives proportions between limits

    """

    likelihood: Likelihood
    prior: Prior

    @property
    def args(self) -> tuple:
        return (self.likelihood.args, self.prior.args)

    @property
    def mass_func(self) -> Callable:
        interval_func = self.likelihood.interval_func
        between_func = self.prior.between_func
        return _model_mass(interval_func, between_func)

    @property
    def integrate_func(self) -> Callable:
        return _quad_bound.integrator(self.mass_func)

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
        if not rtol >= 1e-7:
            raise ValueError(rtol)

        return self.integrate_func(self.args, rtol)

    def mass(self, ratio: float) -> float:
        """Return the prior mass inside the interval at likelihood ratio."""
        lo, hi = self.likelihood.interval(ratio)
        return self.prior.between(lo, hi)


@functools.lru_cache(maxsize=None)
def _model_mass(interval_func, between_func):
    @numba.njit
    def mass(args, ratio):
        args_interval, args_between = args
        lo, hi = interval_func(args_interval, ratio)
        return between_func(args_between, lo, hi)

    return mass
