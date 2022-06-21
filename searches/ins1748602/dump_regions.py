"""
Usage:

python searches/ins1748602/dump_regions.py

"""

import os

import pyhf

from pyhf_stuff import region, serial

BASEPATH = os.path.dirname(__file__)


def main():
    for name, sr_name, workspace in generate_regions():
        region.Region(sr_name, workspace).dump(
            os.path.join(BASEPATH, name),
        )


def generate_regions():
    # a
    spec = serial.load_json_gz(os.path.join(BASEPATH, "a_bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
    channels = workspace.channel_slices.keys()

    # find control and signal region strings
    control_regions = set()
    signal_regions = set()
    for name in channels:
        if name.startswith("CR"):
            control_regions.add(name)
            continue

        if name.startswith("VR"):
            continue

        assert name.startswith("SR")
        signal_regions.add(name)

    (sr_name,) = signal_regions
    bins = range(workspace.channel_nbins[sr_name])
    workspace = region.prune(workspace, sr_name, *control_regions)
    workspace = region.merge_to_bins(workspace, sr_name, bins)
    yield "SRA", sr_name, workspace

    # b
    spec = serial.load_json_gz(os.path.join(BASEPATH, "b_bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
    channels = workspace.channel_slices.keys()

    # find control and signal region strings
    control_regions = set()
    signal_regions = set()
    for name in channels:
        if name.startswith("CR"):
            control_regions.add(name)
            continue

        if name.startswith("VR"):
            continue

        assert name.startswith("SR")
        signal_regions.add(name)

    (sr_name,) = signal_regions
    assert workspace.channel_nbins[sr_name] == 1
    workspace = region.prune(workspace, sr_name, *control_regions)
    yield "SRB", sr_name, workspace

    # c
    spec = serial.load_json_gz(os.path.join(BASEPATH, "c_bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
    channels = workspace.channel_slices.keys()

    # find control and signal region strings
    control_regions = set()
    signal_regions = set()
    for name in channels:
        if name.startswith("CR"):
            control_regions.add(name)
            continue

        if name.startswith("VR"):
            continue

        assert name.startswith("SR")
        signal_regions.add(name)

    (sr_name,) = signal_regions
    bins = range(workspace.channel_nbins[sr_name])
    workspace = region.prune(workspace, sr_name, *control_regions)
    workspace = region.merge_to_bins(workspace, sr_name, bins)
    yield "SRC", sr_name, workspace


if __name__ == "__main__":
    main()
