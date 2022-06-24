"""Scan for optima at fixed signal region yields."""
import os
from dataclasses import asdict, dataclass

import jax
import numpy
import scipy

from . import serial
from .fit_interval import _suppress_bounds_warning
from .region_fit import region_fit
from .region_properties import region_properties


def fit(region, start, stop, num):

    first_value_and_grad = _fit_first_value_and_grad(region)

    levels = []
    for yield_ in numpy.linspace(start, stop, num):
        # slsqp results depend on initialization (stuck in local minima?)
        # use suggested init and alternatives and take the best
        optimum_from_suggested = _fit_slsqp(region, yield_)

        first = _fit_soft_constraint(region, first_value_and_grad, yield_)
        optimum_from_soft = _fit_slsqp(region, yield_, init=first.x)

        optimum_from_fit = _fit_slsqp(
            region, yield_, init=region_fit(region).x
        )

        optimum = min(
            [
                optimum_from_suggested,
                optimum_from_soft,
                optimum_from_fit,
            ],
            key=lambda x: x.fun,
        )
        if not optimum.success:
            raise RuntimeError(yield_)

        levels.append(optimum.fun)

    return FitLinspace(
        start=start,
        stop=stop,
        levels=levels,
    )


def _fit_slsqp(region, yield_, *, init=None):
    properties = region_properties(region)

    if init is None:
        init = properties.init

    constaint = scipy.optimize.NonlinearConstraint(
        properties.yield_value,
        yield_,
        yield_,
        jac=properties.yield_grad,
    )

    with _suppress_bounds_warning():
        optimum = scipy.optimize.minimize(
            properties.objective_value_and_grad,
            init,
            bounds=properties.bounds,
            jac=True,
            method="SLSQP",
            constraints=constaint,
            options=dict(maxiter=10_000, ftol=1e-6),
        )
    return optimum


def _fit_soft_constraint(region, first_value_and_grad, yield_):
    properties = region_properties(region)
    # initial estimate from constrained minimization
    return scipy.optimize.minimize(
        lambda x: first_value_and_grad(x, yield_),
        properties.init,
        bounds=properties.bounds,
        jac=True,
        method="L-BFGS-B",
    )


def _fit_first_value_and_grad(region, tol=1e-2):
    properties = region_properties(region)

    def objective(x, yield_):
        inv_var = jax.numpy.maximum(tol, tol * yield_) ** -2
        constraint = inv_var * 0.5 * (properties.yield_value(x) - yield_) ** 2
        return properties.objective_value(x) + constraint

    return jax.jit(jax.value_and_grad(objective))


# serialization


@dataclass(frozen=True)
class FitLinspace:
    start: float
    stop: float
    levels: list[float]

    filename = "linspace"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
