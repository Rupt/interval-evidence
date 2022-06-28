"""
Usage:

python searches/ins1750597/dump_regions.py

"""

import os

import pyhf

from pyhf_stuff import region, serial

BASEPATH = os.path.dirname(__file__)


def main():
    for sr_name, workspace in generate_regions():
        region.Region(sr_name, workspace).dump(
            os.path.join(BASEPATH, region.strip_cuts(sr_name)),
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

    print(sorted(control_regions))
    print(sorted(signal_regions))

    sr_name = "SRtest"
    workspace_i = region.merge_channels(
        workspace,
        sr_name,
        [
            "SRDF_0a_cuts",
            "SRDF_0b_cuts",
        ],
    )
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    return
    # (sr_name,) = signal_regions
    # bins = range(workspace.channel_nbins[sr_name])
    # workspace = region.prune(workspace, [sr_name, *control_regions])
    # workspace = region.merge_to_bins(workspace, sr_name, bins)
    # yield "SRA", sr_name, workspace


if __name__ == "__main__":
    main()
