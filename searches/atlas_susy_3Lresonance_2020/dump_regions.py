"""
Usage:

python searches/atlas_susy_3Lresonance_2020/dump_regions.py

"""
import os

import pyhf

from discohisto import region, serial

BASEPATH = os.path.dirname(__file__)


def main():
    for name, workspace in generate_regions():
        region.Region(name, workspace).dump(
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

        assert name.startswith("SR")
        signal_regions.add(name)

    # serialize regions for each
    for name in sorted(signal_regions):
        workspace_i = region.prune(workspace, [name, *control_regions])
        # these workspaces have bugs; remove them
        workspace_i = region.filter_modifiers(
            workspace_i,
            [
                empty_shapesys,
                empty_histosys,
            ],
        )
        yield name, workspace_i


def empty_shapesys(modifier, sample, channel):
    # error: poisson([0.0]) shapesys
    return modifier["type"] == "shapesys" and modifier["data"] == [0.0]


def empty_histosys(modifier, sample, channel):
    # error: histosys with no variation
    if modifier["type"] != "histosys":
        return False
    data = modifier["data"]
    return data["hi_data"] == data["lo_data"] == sample["data"]


if __name__ == "__main__":
    main()
