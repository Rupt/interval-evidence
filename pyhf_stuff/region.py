"""Regions are single-signal-region workspaces. Define utilities."""
import os
from dataclasses import dataclass

import pyhf

from . import serial


@dataclass(frozen=True)
class Region:
    signal_region_name: str
    workspace: pyhf.Workspace

    def __post_init__(self):
        if self.signal_region_name not in self.workspace.channel_slices:
            raise ValueError(self.signal_region_name)


# serialization

FILENAME = "region.json.gz"


def dump(region, path):
    os.makedirs(path, exist_ok=True)

    region_json = {
        "signal_region_name": region.signal_region_name,
        "workspace": region.workspace,
    }

    serial.dump_json_gz(region_json, os.path.join(path, FILENAME))


def load(path):
    region_json = serial.load_json_gz(os.path.join(path, FILENAME))

    return Region(
        region_json["signal_region_name"],
        pyhf.Workspace(region_json["workspace"]),
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
    """Prune to keep only channel names given in args."""
    remove = workspace.channel_slices.keys() - args
    return workspace.prune(channels=remove)
