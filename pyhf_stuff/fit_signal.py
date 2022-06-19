"""Scan for optima at fixed additive signal contributions."""
import os
from dataclasses import asdict, dataclass
from typing import List

import jax
import numpy
import pyhf
import scipy

from . import serial
from .region_properties import region_properties

FILENAME = "signal.json"


def fit(region, start, stop, num):
    properties = region_properties(region)

    # negative signals are nonsense
    if not start >= 0:
        raise ValueError(start)

    (ndata,) = properties.data[properties.slice_]

    def objective(x, signal):
        # signal region likelihood is poisson(n | background + signal)
        background = properties.yield_value(x)
        # using pyhf poisson for consistency
        logl = pyhf.probability.Poisson(background + signal).log_prob(ndata)
        return properties.objective_value(x) - logl

    objective_and_grad = jax.jit(jax.value_and_grad(objective))

    def optimum_given_signal(signal):
        optimum = scipy.optimize.minimize(
            lambda x: objective_and_grad(x, signal),
            properties.init,
            bounds=properties.bounds,
            jac=True,
            method="L-BFGS-B",
        )
        assert optimum.success
        return optimum.fun

    levels = [
        optimum_given_signal(yield_)
        for yield_ in numpy.linspace(start, stop, num)
    ]

    return FitSignal(
        start=start,
        stop=stop,
        levels=levels,
    )


# serialization


@dataclass(frozen=True)
class FitSignal:
    start: float
    stop: float
    levels: List[float]

    def dump(self, path):
        os.makedirs(path, exist_ok=True)
        serial.dump_json_human(asdict(self), os.path.join(path, FILENAME))

    @classmethod
    def load(cls, path):
        obj_json = serial.load_json(os.path.join(path, FILENAME))
        return cls(**obj_json)
