"""Regions are single-signal-region workspaces."""
import os
from dataclasses import dataclass

import pyhf

from . import serial

FILENAME = "region.json.gz"


@dataclass(frozen=True, eq=False)
class Region:
    signal_region_name: str
    workspace: pyhf.Workspace

    def __post_init__(self):
        if self.signal_region_name not in self.workspace.channel_slices:
            raise ValueError(self.signal_region_name)

    @property
    def ndata(self) -> int:
        (data,) = self.workspace.observations[self.signal_region_name]
        assert data == int(data)
        return int(data)

    # avoid hashing the spooky scary dicts inside us
    def __hash__(self):
        return object.__hash__(self)

    # serialization
    def dump(self, path):
        os.makedirs(path, exist_ok=True)

        region_json = {
            "signal_region_name": self.signal_region_name,
            "workspace": self.workspace,
        }

        serial.dump_json_gz(region_json, os.path.join(path, FILENAME))

    @classmethod
    def load(cls, path):
        region_json = serial.load_json_gz(os.path.join(path, FILENAME))

        return cls(
            signal_region_name=region_json["signal_region_name"],
            workspace=pyhf.Workspace(region_json["workspace"]),
        )


# utilities


def strip_cuts(name, *, cuts="_cuts"):
    if name.endswith(cuts):
        return name[: -len(cuts)]
    return name


def clear_poi(spec):
    """Set all measurement poi in spec to the empty string.

    This avoids exceptions thrown by pyhf workspace stuff.
    """
    for measurement in spec["measurements"]:
        measurement["config"]["poi"] = ""
    return spec


def prune(workspace, *args):
    """Prune to keep only channel names given in region_names."""
    remove = workspace.channel_slices.keys() - args
    return workspace.prune(channels=remove)
