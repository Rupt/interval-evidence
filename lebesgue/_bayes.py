"""Manage likelihoods and priors."""
from dataclasses import dataclass

import numba
from numba.core.types.misc import ClassInstanceType

from . import _core
from ._quad_bound import _quad_bound

# python-facing


@dataclass(frozen=True)
class Likelihood:
    _likelihood: ClassInstanceType

    def __post_init__(self):
        type_ = numba.typeof(self._likelihood)
        if not isinstance(type_, ClassInstanceType):
            raise TypeError(self._likelihood)

        jit_methods = type_.jit_methods
        if "_interval" not in jit_methods:
            raise TypeError(jit_methods.keys())

    def interval(self, ratio):
        ratio = float(ratio)
        assert 0 <= ratio <= 1, ratio
        return self._likelihood._interval(ratio)


@dataclass(frozen=True)
class Prior:
    _prior: ClassInstanceType

    def __post_init__(self):
        type_ = numba.typeof(self._prior)
        if not isinstance(type_, ClassInstanceType):
            raise TypeError(self._prior)

        jit_methods = type_.jit_methods
        if "_between" not in jit_methods:
            raise TypeError(jit_methods.keys())

    def between(self, lo, hi):
        lo = float(lo)
        hi = float(hi)
        assert lo <= hi, (lo, hi)
        return self._prior._between(lo, hi)


@dataclass(frozen=True, init=False)
class Model:
    likelihood: Likelihood
    prior: Prior

    _model: ClassInstanceType

    def __init__(self, likelihood, prior):
        if not isinstance(likelihood, Likelihood):
            raise TypeError(likelihood)

        if not isinstance(prior, Prior):
            raise TypeError(prior)

        cls = _model_class(
            numba.typeof(likelihood._likelihood),
            numba.typeof(prior._prior),
        )
        _model = cls(likelihood._likelihood, prior._prior)

        object.__setattr__(self, "likelihood", likelihood)
        object.__setattr__(self, "prior", prior)
        object.__setattr__(self, "_model", _model)

    def integrate(self, *, rtol=1e-2):
        rtol = float(rtol)
        # Tiny tol (< 2 ** -51?) can cause runaway recursion.
        # Small tol is slow and unlikely to be useful.
        assert rtol >= 1e-7, rtol

        return _quad_bound(self._model, rtol)

    def mass(self, ratio):
        lo, hi = self.likelihood.interval(ratio)
        return self.prior.between(lo, hi)


# numba-facing


@_core.cache
def _model_class(likelihood_type, prior_type):
    @_core.jitclass
    class _Model:
        likelihood: likelihood_type
        prior: prior_type

        def __init__(self, likelihood, prior):
            self.likelihood = likelihood
            self.prior = prior

        def _mass(self, ratio):
            lo, hi = self.likelihood._interval(ratio)
            return self.prior._between(lo, hi)

    return _Model
