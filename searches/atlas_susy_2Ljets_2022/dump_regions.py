"""
Usage:

python searches/atlas_susy_2Ljets_2022/dump_regions.py

"""

import os

import numpy
import pyhf

from discohisto import region, serial

BASEPATH = os.path.dirname(__file__)


def main():
    for name, sr_name, workspace in generate_regions():
        region.Region(sr_name, workspace).dump(
            os.path.join(BASEPATH, name),
        )


def generate_regions():
    def ewk(filename):
        spec = serial.load_json_gz(
            os.path.join(BASEPATH, filename + "_bkg.json.gz")
        )
        ensure_true_zeros(spec)
        workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
        channels = workspace.channel_slices.keys()

        control_regions = set()
        signal_regions = set()
        for name in channels:
            if not name.startswith("DR"):
                control_regions.add(name)
                continue

            signal_regions.add(name)

        (sr_name,) = signal_regions
        return filename, sr_name, prune_discovery(workspace)

    for reg in ["high", "llbb", "int", "low", "offshell"]:
        yield ewk("ewk_" + reg)

    def rjr(filename):
        spec = serial.load_json_gz(
            os.path.join(BASEPATH, filename + "_bkg.json.gz")
        )
        workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
        channels = workspace.channel_slices.keys()

        control_regions = set()
        signal_regions = set()
        for name in channels:
            if name.startswith("VR"):
                continue

            if not name.startswith("SR"):
                control_regions.add(name)
                continue

            signal_regions.add(name)

        (sr_name,) = signal_regions
        workspace = region.prune(workspace, [sr_name, *control_regions])
        return filename, sr_name, prune_discovery(workspace)

    yield rjr("rjr_sr2l_low")
    yield rjr("rjr_sr2l_isr")

    def strong(filename):
        spec = serial.load_json_gz(
            os.path.join(BASEPATH, filename + "_bkg.json.gz")
        )
        workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
        channels = workspace.channel_slices.keys()

        control_regions = set()
        signal_regions = set()
        for name in channels:
            if not name.startswith("SR"):
                control_regions.add(name)
                continue

            signal_regions.add(name)

        (sr_name,) = signal_regions
        # no pruning needed
        return filename, sr_name, prune_discovery(workspace)

    src = ["src_12_31", "src_12_61", "src_31_81", "src_81"]
    srhigh = ["srhigh_12_301", "srhigh_301"]
    srlow = ["srlow_12_81", "srlow_101_201", "srlow_101_301", "srlow_301"]
    srmed = ["srmed_12_101", "srmed_101"]
    srz = ["srzhigh", "srzlow", "srzmed"]

    for reg in src + srhigh + srlow + srmed + srz:
        yield strong("str_" + reg)


def prune_discovery(workspace):
    return workspace.prune(
        modifiers=["mu_Discovery"],
        samples=["DiscoveryMode_Discovery"],
    )


def ensure_true_zeros(spec):
    # something has replaced zero with a small float; fix it
    should_be_zero = float(numpy.float32(1e-4))
    for observation in spec["observations"]:
        if observation["data"] == [should_be_zero]:
            observation["data"] = [0]


if __name__ == "__main__":
    main()
