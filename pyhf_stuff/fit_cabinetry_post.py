import os
from dataclasses import asdict, dataclass

from . import serial
from .fit_cabinetry import _fit
from .region_properties import region_properties


def fit(region):
    properties = region_properties(region)
    # unblinded -> post-fit
    yield_post, error_post = _fit(
        properties.model,
        properties.data,
        region.signal_region_name,
    )
    return FitCabinetryPost(
        yield_post=yield_post,
        error_post=error_post,
    )


# serialization


@dataclass(frozen=True)
class FitCabinetryPost:
    yield_post: float
    error_post: float

    filename = "cabinetry_post"

    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)
        filename = self.filename + suffix + ".json"
        serial.dump_json_human(asdict(self), os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json"
        obj_json = serial.load_json(os.path.join(path, filename))
        return cls(**obj_json)
