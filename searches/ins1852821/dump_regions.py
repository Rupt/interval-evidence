"""
Usage:

python searches/ins1852821/dump_regions.py

"""

import os

import pyhf

from pyhf_stuff import region, serial

BASEPATH = os.path.dirname(__file__)


def main():
    for name, workspace in generate_regions():
        region.Region(name, 0, workspace).dump(
            os.path.join(BASEPATH, region.strip_cuts(name)),
        )


def generate_regions():
    spec = serial.load_json_gz(os.path.join(BASEPATH, "bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))

    channels = workspace.channel_slices.keys()

    # find control and signal region strings
    control_regions = set()
    signal_regions = set()
    for name in channels:
        if name.startswith("CR"):
            control_regions.add(name)
            continue

        # ins1852821 modifies its control regions to exclude 5-lepton events
        # but we don't have specs for those control regions
        if name.startswith("SR5L"):
            continue

        assert name.startswith("SR")
        signal_regions.add(name)

    # serialize regions for each
    for name in sorted(signal_regions):
        yield name, region.prune(workspace, name, *control_regions)


if __name__ == "__main__":
    main()
