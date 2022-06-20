import os
from dataclasses import asdict, dataclass

import cabinetry
import numpy

from . import serial
from .region_properties import region_properties


def fit(region):
    properties = region_properties(region)
    # blinded -> pre-fit
    yield_pre, error_pre = _fit(
        properties.model_blind,
        properties.data,
        region.signal_region_name,
        region.signal_region_bins,
    )
    return FitCabinetry(
        yield_pre=yield_pre,
        error_pre=error_pre,
    )


def _fit(model, data, region_name, bins):
    prediction = cabinetry.model_utils.prediction(
        model, fit_results=cabinetry.fit.fit(model, data)
    )
    index = model.config.channels.index(region_name)
    yield_ = numpy.array(prediction.model_yields)[index, :, list(bins)].sum()

    nbins = len(prediction.total_stdev_model_bins[index])
    if nbins == len(bins):
        # we have all bins - use total
        error = prediction.total_stdev_model_channels
    else:
        # naive combination :(
        errors = prediction.total_stdev_model_bins[index][list(bins)]
        error = errors.dot(errors) ** 0.5

    return yield_, error


# serialization


@dataclass(frozen=True)
class FitCabinetry:
    yield_pre: float
    error_pre: float

    filename = "cabinetry"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
