"""
Usage:

python searches/ins1765529/dump_regions.py

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
    # low
    spec = serial.load_json_gz(os.path.join(BASEPATH, "low_mass_bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
    channels = workspace.channel_slices.keys()

    # find control and signal region strings
    control_regions = set()
    signal_regions = set()
    for name in channels:
        if not name.startswith("SR"):
            control_regions.add(name)
            continue

        signal_regions.add(name)

    (sr_name,) = signal_regions
    name = "SRlowMass"
    yield name, sr_name, region.prune(workspace, sr_name, *control_regions)

    # high
    spec = serial.load_json_gz(os.path.join(BASEPATH, "high_mass_bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
    channels = workspace.channel_slices.keys()

    # find control and signal region strings
    control_regions = set()
    signal_regions = set()
    for name in channels:
        if not name.startswith("SR"):
            control_regions.add(name)
            continue

        signal_regions.add(name)

    (sr_name,) = signal_regions
    name = "SRhighMass"
    yield name, sr_name, region.prune(workspace, sr_name, *control_regions)


if __name__ == "__main__":
    main()
