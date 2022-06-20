"""
Usage:

python searches/ins1788448/dump_regions.py

"""

import glob
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
    paths = glob.glob(os.path.join(BASEPATH, "*_bkg.json.gz"))

    for path in paths:
        spec = serial.load_json_gz(path)
        workspace = pyhf.workspace.Workspace(region.clear_poi(spec))

        channels = workspace.channel_slices.keys()

        # find control and signals region strings
        control_regions = set()
        signal_regions = set()
        for name in channels:
            if not name.startswith("SR"):
                control_regions.add(name)
                continue

            assert name.startswith("SR")
            signal_regions.add(name)

        (name,) = signal_regions

        # serialize regions for each
        yield name, region.prune(workspace, name, *control_regions)


if __name__ == "__main__":
    main()
