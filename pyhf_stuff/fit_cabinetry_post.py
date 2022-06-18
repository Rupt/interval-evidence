import os
from dataclasses import asdict, dataclass

from . import serial
from .fit_cabinetry import _fit
from .region_properties import region_properties

FILENAME = "cabinetry_post.json"


def fit(region):
    properties = region_properties(region)
    # unblinded -> post-fit
    yield_post, error_post = _fit(
        properties.model, properties.data, region.signal_region_name
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

    def dump(self, path):
        os.makedirs(path, exist_ok=True)
        serial.dump_json_human(asdict(self), os.path.join(path, FILENAME))

    @classmethod
    def load(cls, path):
        obj_json = serial.load_json(os.path.join(path, FILENAME))
        return cls(**obj_json)
