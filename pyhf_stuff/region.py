"""Regions are single-signal-region workspaces."""
import copy
import os
from dataclasses import dataclass, field

import pyhf

from . import serial


@dataclass(frozen=True, eq=False)
class Region:
    signal_region_name: str
    workspace: pyhf.Workspace

    _cache: dict = field(
        default_factory=dict, init=False, hash=False, compare=False
    )

    filename = "region"

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
    def dump(self, path, *, suffix=""):
        os.makedirs(path, exist_ok=True)

        region_json = {
            "signal_region_name": self.signal_region_name,
            "workspace": self.workspace,
        }

        filename = self.filename + suffix + ".json.gz"
        serial.dump_json_gz(region_json, os.path.join(path, filename))

    @classmethod
    def load(cls, path, *, suffix=""):
        filename = cls.filename + suffix + ".json.gz"
        region_json = serial.load_json_gz(os.path.join(path, filename))

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


def prune(workspace, channel_names_to_keep):
    """Return a workspace keeping only channel names given in args."""
    remove = workspace.channel_slices.keys() - channel_names_to_keep
    return workspace.prune(channels=remove)


def merge_to_bins(workspace, channel_name, bins):
    """Return a workspace with channel bins combined into a signle bin."""
    bins = list(bins)

    # see https://pyhf.readthedocs.io/en/v0.6.3/likelihood.html#modifiers
    def combine(a):
        return sum(a[i] for i in bins)

    def dot(a, b):
        return sum(a[i] * b[i] for i in bins)

    def merge_modifier(modifier):
        type_ = modifier["type"]
        data = modifier["data"]
        if type_ == "staterror":
            # sum stat errors in quadrature
            new_data = [dot(data, data) ** 0.5]
            return dict(modifier, data=new_data)
        if type_ == "histosys":
            # sum high and low parts
            return dict(
                modifier,
                data=dict(
                    hi_data=[combine(data["hi_data"])],
                    lo_data=[combine(data["lo_data"])],
                ),
            )
        if type_ in ("normsys", "lumi", "normfactor"):
            # normsys, lumi apply equally to all bins
            return modifier
        # not sure about "shapefactor"; I've seen no examples
        raise NotImplementedError(type_)

    def merge_channel(channel):
        if channel["name"] != channel_name:
            return channel
        return {
            "name": channel["name"],
            "samples": [
                {
                    "name": sample["name"],
                    "data": [combine(sample["data"])],
                    "modifiers": [
                        merge_modifier(modifier)
                        for modifier in sample["modifiers"]
                    ],
                }
                for sample in channel["samples"]
            ],
        }

    def merge_observation(observation):
        if observation["name"] != channel_name:
            return observation
        return dict(observation, data=[combine(observation["data"])])

    newspec = {
        "channels": [
            merge_channel(channel) for channel in workspace["channels"]
        ],
        # measurements are unchanged
        "measurements": workspace["measurements"],
        "observations": [
            merge_observation(observation)
            for observation in workspace["observations"]
        ],
        "version": workspace["version"],
    }
    return pyhf.Workspace(newspec)


def filter_modifiers(workspace, filters):
    newspec = copy.deepcopy(dict(workspace))

    filters = list(filters)

    def filter_(modifier, sample, channel):
        return any(filt(modifier, sample, channel) for filt in filters)

    for channel in newspec["channels"]:
        for sample in channel["samples"]:
            good = []
            for modifier in sample["modifiers"]:
                if filter_(modifier, sample, channel):
                    continue
                good.append(modifier)
            sample["modifiers"] = good

    return pyhf.Workspace(newspec)
