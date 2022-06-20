"""
Usage:

python searches/ins1827025/dump_regions.py

"""

import glob
import os

import pyhf

from pyhf_stuff import region, serial

BASEPATH = os.path.dirname(__file__)
SR_NAME = "SR_cuts"


def main():
    for name, workspace in generate_regions():
        region.Region(SR_NAME, 0, workspace).dump(
            os.path.join(BASEPATH, name),
        )


def generate_regions():
    paths = glob.glob(os.path.join(BASEPATH, "*.json.gz"))

    for path in paths:
        spec = serial.load_json_gz(path)
        workspace = pyhf.workspace.Workspace(region.clear_poi(spec))

        channels = workspace.channel_slices.keys()

        # find control and signals region strings
        control_regions = set()
        signal_regions = set()
        for name in channels:
            if name.startswith("CR"):
                control_regions.add(name)
                continue

            assert name.startswith("SR")
            signal_regions.add(name)

        (sr_name,) = signal_regions
        assert sr_name == SR_NAME

        name = os.path.basename(path).split("_bkgonly.json.gz")[0]
        yield name, region.prune(workspace, sr_name, *control_regions)


if __name__ == "__main__":
    main()
