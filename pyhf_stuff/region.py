"""Regions are single-signal-region workspaces. Define utilities."""
import os

from . import serial

# serialization

FILENAME = "region.json.gz"


def dump(signal_region_name, workspace, path):
    os.makedirs(path, exist_ok=True)

    region = {
        "signal_region_name": signal_region_name,
        "workspace": workspace,
    }

    serial.dump_json_gz(
        region,
        os.path.join(path, FILENAME),
    )


def load(path):
    region = serial.load_json_gz(os.path.join(path, FILENAME))

    signal_region_name = region["signal_region_name"]
    workspace = region["workspace"]

    return signal_region_name, workspace


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
