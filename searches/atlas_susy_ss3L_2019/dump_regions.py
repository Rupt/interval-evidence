"""
Usage:

python searches/atlas_susy_ss3L_2019/dump_regions.py

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
    def standard(filename):
        spec = serial.load_json_gz(os.path.join(BASEPATH, filename))
        workspace = pyhf.workspace.Workspace(region.clear_poi(spec))
        (name,) = workspace.channel_slices.keys()
        return name, workspace

    # rpc2l0b
    yield standard("rpc2l0b_bkg.json.gz")

    # rpc2l1b
    yield standard("rpc2l1b_bkg.json.gz")

    # rpc2l2b
    yield standard("rpc2l2b_bkg.json.gz")

    # rpc3lss1b
    yield standard("rpc3lss1b_bkg.json.gz")

    # rpv2l
    yield standard("rpv2l_bkg.json.gz")


if __name__ == "__main__":
    main()
