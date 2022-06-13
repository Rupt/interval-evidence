import os
from dataclasses import asdict, dataclass

import numpy
import scipy

from . import serial
from .region_properties import region_properties

FILENAME = "normal.json"


def fit(region):
    properties = region_properties(region)

    optimum = scipy.optimize.minimize(
        properties.objective_value_and_grad,
        properties.init,
        bounds=properties.bounds,
        jac=True,
        method="L-BFGS-B",
    )
    assert optimum.success

    # if gaussian, then covariance is inverse hessian
    cov = properties.objective_hess_inv(optimum.x)

    # approximate signal region yield (log) linearly from gradients
    yield_value, yield_grad = properties.yield_value_and_grad(optimum.x)

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


def dump(fit: FitNormal, path):
    os.makedirs(path, exist_ok=True)
    serial.dump_json_human(asdict(fit), os.path.join(path, FILENAME))


def load(path) -> FitNormal:
    obj_json = serial.load_json(os.path.join(path, FILENAME))
    return FitNormal(**obj_json)
