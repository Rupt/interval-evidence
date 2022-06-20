"""Fit for levels of fixed values.

This fit is unreliable.
Its optimizations often fail to converge and return incorrect results.
Until that is fixed, I recommend interpolating with fit_linspace.

"""
import os
import warnings
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import scipy

from . import serial, stats
from .region_properties import region_properties

DEFAULT_LEVELS = tuple(stats.sigma_to_llr(range(1, 6 + 1)))


def fit(region, *, levels=DEFAULT_LEVELS):
    properties = region_properties(region)

    optimum = scipy.optimize.minimize(
        properties.objective_value_and_grad,
        properties.init,
        bounds=properties.bounds,
        jac=True,
        method="L-BFGS-B",
    )
    assert optimum.success

    def minmax_given_level(level):
        constaint = scipy.optimize.NonlinearConstraint(
            properties.objective_value,
            optimum.fun + level,
            optimum.fun + level,
            jac=properties.objective_grad,
        )

        # some upper bounds have failed with default maximum iterations (100)
        with _suppress_bounds_warning():
            minimum = scipy.optimize.minimize(
                properties.yield_value_and_grad,
                properties.init,
                bounds=properties.bounds,
                jac=True,
                method="SLSQP",
                constraints=constaint,
                options=dict(maxiter=1000),
            )
        assert minimum.success

        def negative_yield_value_and_grad(x):
            value, grad = properties.yield_value_and_grad(x)
            return -value, -grad

        with _suppress_bounds_warning():
            maximum = scipy.optimize.minimize(
                negative_yield_value_and_grad,
                properties.init,
                bounds=properties.bounds,
                jac=True,
                method="SLSQP",
                constraints=constaint,
                options=dict(maxiter=1000),
            )
        assert maximum.success

        return [minimum.fun, -maximum.fun]

    intervals = [minmax_given_level(level) for level in levels]

    return FitInterval(
        levels=list(levels),
        intervals=intervals,
    )


@contextmanager
def _suppress_bounds_warning():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            (
                "Values in x were outside bounds during a "
                "minimize step, clipping to bounds"
            ),
            category=RuntimeWarning,
        )
        yield


# serialization


@dataclass(frozen=True)
class FitInterval:
    levels: list[float]
    intervals: list[list[float]]

    filename = "interval"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
