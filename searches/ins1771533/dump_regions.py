"""
Usage:

python searches/ins1771533/dump_regions.py

"""

import os

import pyhf

from pyhf_stuff import region, serial

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

    sr_low = "SRlow_cuts"
    sr_isr = "SRISR_cuts"
    cr_low = "CRlow_cuts"
    cr_isr = "CRISR_cuts"

    assert control_regions == {cr_low, cr_isr}
    assert signal_regions == {sr_low, sr_isr}

    yield sr_low, region.prune(workspace, sr_low, cr_low)
    yield sr_isr, region.prune(workspace, sr_isr, sr_isr)


if __name__ == "__main__":
    main()
