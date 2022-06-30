import os
from dataclasses import asdict, dataclass

import numpy

from . import serial
from .region_fit import region_fit
from .region_properties import region_properties


def fit(region):
    properties = region_properties(region)

    optimum_x = region_fit(region).x

    # if gaussian, then covariance is inverse hessian
    cov = properties.objective_hess_inv(optimum_x)

    # approximate signal region yield (log) linearly from gradients
    yield_value, yield_grad = properties.yield_value_and_grad(optimum_x)

    yield_std = _quadratic_form(cov, yield_grad) ** 0.5

    # d/dx log(f(x)) = f'(x) / f(x)
    yield_log_std = _quadratic_form(cov, yield_grad / yield_value) ** 0.5

    # values are 0-dim jax arrays; cast to floats
    return FitNormal(
        yield_linear=float(yield_value),
        error_linear=float(yield_std),
        error_log=float(yield_log_std),
    )


def _quadratic_form(matrix, vector):
    return matrix.dot(vector).dot(vector)


# serialization


@dataclass(frozen=True)
class FitNormal:
    yield_linear: float
    error_linear: float
    error_log: float

    @property
    def yield_log(self):
        return numpy.log(self.yield_linear)

    filename = "normal"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
