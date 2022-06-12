"""Scan for optima at fixed signal region yields."""
import os
from dataclasses import asdict, dataclass
from typing import List

import numpy
import scipy

from . import serial
from .region_properties import region_properties

FILENAME = "linspace.json"


def fit(region, start, stop, num):
    properties = region_properties(region)

    def optimum_given_yield(yield_):
        constaint = scipy.optimize.NonlinearConstraint(
            properties.yield_value,
            yield_,
            yield_,
            jac=properties.yield_grad,
        )

        optimum = scipy.optimize.minimize(
            properties.objective_value_and_grad,
            properties.init,
            bounds=properties.bounds,
            jac=True,
            method="SLSQP",
            constraints=constaint,
        )
        assert optimum.success

        return optimum.fun

    levels = [
        optimum_given_yield(yield_)
        for yield_ in numpy.linspace(start, stop, num)
    ]

    return FitLinspace(
        start=start,
        stop=stop,
        levels=levels,
    )


# serialization


@dataclass(frozen=True)
class FitLinspace:
    start: float
    stop: float
    levels: List[float]


def dump(fit: FitLinspace, path):
    os.makedirs(path, exist_ok=True)
    serial.dump_json_human(asdict(fit), os.path.join(path, FILENAME))


def load(path) -> FitLinspace:
    obj_json = serial.load_json(os.path.join(path, FILENAME))
    return FitLinspace(**obj_json)
