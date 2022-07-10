"""Scan for optima at fixed signal region yields."""
import os
from dataclasses import asdict, dataclass

import numpy
import scipy

from . import serial
from .fit_interval import _suppress_bounds_warning
from .region_properties import region_properties


def fit(region, start, stop, num, *, anchors=None):
    if anchors is None:
        anchors = []
    anchors = list(anchors)

    anchor_inits = []

    def fit_optimum(region, yield_):
        # slsqp results depend on initialization (stuck in local minima?)
        # use suggested init and alternatives and take the best
        optimum_from_suggested = _fit_slsqp(region, yield_)
        optima_from_anchors = (
            _fit_slsqp(region, yield_, init=x) for x in anchor_inits
        )
        return min(
            [optimum_from_suggested, *optima_from_anchors],
            key=lambda x: x.fun,
        )

    # populate anchors, bootstrapping off each as we go
    for anchor in anchors:
        anchor_inits.append(fit_optimum(region, anchor).x)

    levels = []
    for yield_ in numpy.linspace(start, stop, num):
        optimum = fit_optimum(region, yield_)
        if not optimum.success:
            raise RuntimeError(yield_)

        levels.append(optimum.fun)

    return FitLinspace(
        start=start,
        stop=stop,
        anchors=anchors,
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
            options=dict(maxiter=15_000),
        )
    return optimum


# serialization


@dataclass(frozen=True)
class FitLinspace:
    start: float
    stop: float
    anchors: list[float]
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
