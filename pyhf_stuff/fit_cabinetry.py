import os
from dataclasses import asdict, dataclass

import cabinetry
import numpy

from . import serial
from .region_properties import region_properties

FILENAME = "cabinetry.json"


def fit(region):
    properties = region_properties(region)
    # blinded -> pre-fit
    yield_pre, error_pre = _fit(
        properties.model_blind, properties.data, region.signal_region_name
    )
    # unblinded -> post-fit
    yield_post, error_post = _fit(
        properties.model, properties.data, region.signal_region_name
    )
    return FitCabinetry(
        yield_pre=yield_pre,
        error_pre=error_pre,
        yield_post=yield_post,
        error_post=error_post,
    )


def _fit(model, data, signal_region_name):
    prediction = cabinetry.model_utils.prediction(
        model, fit_results=cabinetry.fit.fit(model, data)
    )
    index = model.config.channels.index(signal_region_name)
    yield_ = numpy.sum(prediction.model_yields[index])
    error = prediction.total_stdev_model_channels[index]
    return yield_, error


# serialization


@dataclass(frozen=True)
class FitCabinetry:
    yield_pre: float
    error_pre: float
    yield_post: float
    error_post: float


def dump(fit: FitCabinetry, path):
    os.makedirs(path, exist_ok=True)
    serial.dump_json_human(asdict(fit), os.path.join(path, FILENAME))


def load(path) -> FitCabinetry:
    obj_json = serial.load_json(os.path.join(path, FILENAME))
    return FitCabinetry(**obj_json)
