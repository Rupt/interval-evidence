"""Regions are single-signal-region workspaces."""
import copy
import os
from collections import defaultdict
from dataclasses import dataclass, field

import numpy
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


def merge_channels(workspace, name, channels_to_merge):
    """Return a workspace with given channels merged into one."""
    channels_to_merge = set(channels_to_merge)

    # channels: extract samples; since a channel is samples and a name
    # we no longer need channels once we have the contained samples
    chan_merge = []
    channels_new = []
    for chan in workspace["channels"]:
        if chan["name"] in channels_to_merge:
            chan_merge.append(chan)
        else:
            channels_new.append(chan)

    sample_name_to_samps = defaultdict(list)
    for chan in chan_merge:
        for samp in chan["samples"]:
            sample_name_to_samps[samp["name"]].append(samp)

    # per samples of the same name, sum data and combine their modifiers
    channel_samples = []
    for sample_name, samples in sample_name_to_samps.items():
        sample_datas = [samp["data"] for samp in samples]
        sample_data = numpy.sum(sample_datas, axis=0)

        # find the union of all modifiers across channels for this sample
        modifier_name_to_type = {}
        for samp in samples:
            for mod in samp["modifiers"]:
                modifier_name_to_type[mod["name"]] = mod["type"]

        # combine, inserting null defaults where the modifier is missing
        sample_modifiers = []
        for mod_name, mod_type in modifier_name_to_type.items():
            mod_datas = []
            for samp in samples:
                mod = _get_named(samp["modifiers"], mod_name)
                if mod is None:
                    mod_data = _mod_default_data(mod_type, samp)
                else:
                    assert mod["type"] == mod_type, (mod_type, mod)
                    mod_data = mod["data"]
                mod_datas.append(mod_data)

            mod_data = _mod_sum_data(mod_type, mod_datas, sample_datas)

            sample_modifiers.append(
                {
                    "data": mod_data,
                    "name": mod_name,
                    "type": mod_type,
                }
            )

        # we can collect multiple stat errors from the combined channels
        # add them in quadrature
        stat_data = []
        sample_modifiers_other = []
        for mod in sample_modifiers:
            if mod["type"] == "staterror":
                stat_data.append(mod["data"])
            else:
                sample_modifiers_other.append(mod)

        sample_modifiers_other.append(
            {
                "data": _mod_sum_data("staterror", stat_data),
                "name": "staterror_" + name,
                "type": "staterror",
            }
        )
        sample_modifiers = sample_modifiers_other

        channel_samples.append(
            {
                "data": sample_data.tolist(),
                "modifiers": sample_modifiers,
                "name": sample_name,
            }
        )

    channels_new.append(
        {
            "name": name,
            "samples": channel_samples,
        }
    )

    # observations
    obs_merge = []
    observations_new = []
    for obs in workspace["observations"]:
        if obs["name"] in channels_to_merge:
            obs_merge.append(obs)
        else:
            observations_new.append(obs)

    data = numpy.sum([obs["data"] for obs in obs_merge], axis=0)

    observations_new.append(
        {
            "data": data.tolist(),
            "name": name,
        }
    )

    newspec = {
        "channels": channels_new,
        "measurements": workspace["measurements"],
        "observations": observations_new,
        "version": workspace["version"],
    }
    return pyhf.Workspace(newspec)


def _mod_default_data(type_, sample):
    if type_ == "histosys":
        data = sample["data"]
        return {
            "hi_data": data,
            "lo_data": data,
        }
    if type_ == "normsys":
        return {"hi": 1.0, "lo": 1.0}
    if type_ in ("staterror", "shapesys"):
        return [0.0 for _ in sample["data"]]
    if type_ in ("normfactor", "lumi", "shapefactor"):
        return None
    raise NotImplementedError(type_)


def _mod_sum_data(type_, data, sample_data=None):
    if type_ == "histosys":
        hi_data = numpy.sum([data_i["hi_data"] for data_i in data], axis=0)
        lo_data = numpy.sum([data_i["lo_data"] for data_i in data], axis=0)
        return {
            "hi_data": hi_data.tolist(),
            "lo_data": lo_data.tolist(),
        }
    if type_ == "normsys":
        # scale up to data, then normalize back
        norm = numpy.sum(sample_data)
        hi = numpy.sum(
            [
                data_i["hi"] * numpy.array(sample_data_i)
                for data_i, sample_data_i in zip(data, sample_data)
            ]
        )
        lo = numpy.sum(
            [
                data_i["lo"] * numpy.array(sample_data_i)
                for data_i, sample_data_i in zip(data, sample_data)
            ]
        )
        return {
            "hi": float(hi / norm),
            "lo": float(lo / norm),
        }
    if type_ in ("staterror", "shapesys"):
        # sum in quadrature
        return (numpy.sum(numpy.square(data), axis=0) ** 0.5).tolist()
    if type_ in ("normfactor", "lumi", "shapefactor"):
        return None
    raise NotImplementedError(type_)


def _get_named(items, name):
    # awful linear search :(
    # would love to rearrange to a {name: obj} map...
    for item in items:
        if item["name"] == name:
            return item
    return None
