"""
Usage:

python searches/atlas_susy_jets_2021/dump_regions.py

"""

import glob
import os

import pyhf

from discohisto import region, serial

BASEPATH = os.path.dirname(__file__)
SR_NAME = "SR_cuts"


def main():
    for name, workspace in generate_regions():
        region.Region(SR_NAME, workspace).dump(
            os.path.join(BASEPATH, name),
        )


def generate_regions():
    paths = glob.glob(os.path.join(BASEPATH, "*_bkg.json.gz"))

    for path in paths:
        spec = serial.load_json_gz(path)
        workspace = pyhf.workspace.Workspace(region.clear_poi(spec))

        channels = workspace.channel_slices.keys()

        # find control and signal region strings
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

        name = os.path.basename(path).split("_bkg.json.gz")[0]
        yield name, region.prune(workspace, [sr_name, *control_regions])


if __name__ == "__main__":
    main()
