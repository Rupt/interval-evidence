"""
Usage:

python searches/ins1839446/dump_regions.py

"""

import os

import pyhf

from pyhf_stuff import region, serial

BASEPATH = os.path.dirname(__file__)


def main():
    for name, bin_, workspace in generate_regions():
        region.Region(name, bin_, workspace).dump(
            os.path.join(BASEPATH, name),
        )


def generate_regions():
    spec = serial.load_json_gz(os.path.join(BASEPATH, "bkg.json.gz"))
    workspace = pyhf.workspace.Workspace(region.clear_poi(spec))

    channels = workspace.channel_slices.keys()

    # find control and signal region strings
    control_regions_2j = set()
    control_regions_4j = set()
    control_regions_6j = set()
    signal_regions = set()
    for name in channels:
        if name.startswith("WR") or name.startswith("TR"):
            if name[2:4] == "2J":
                control_regions_2j.add(name)
            elif name[2:4] == "4J":
                control_regions_4j.add(name)
            elif name[2:4] == "6J":
                control_regions_6j.add(name)
            else:
                raise ValueError(name)
            continue
        assert name.startswith("SR"), name
        signal_regions.add(name)

    # 2J (gluino) = (SR2JBVEM_meffInc30, 2)
    # 4J high-x   = (SR4JhighxBVEM_meffInc30, 2)
    # 4J low-x    = (SR4JlowxBVEM_meffInc30, 2)
    # 6J (gluino) = (SR6JBVEM_meffInc30, 3)
    # ! no 2J (squark) nor 6J (squark) because they combine bins
    name = "SR2JBVEM_meffInc30"
    assert name in signal_regions
    yield name, 2, region.prune(workspace, name, *control_regions_2j)

    name = "SR4JhighxBVEM_meffInc30"
    assert name in signal_regions
    yield name, 2, region.prune(workspace, name, *control_regions_4j)

    name = "SR4JlowxBVEM_meffInc30"
    assert name in signal_regions
    yield name, 2, region.prune(workspace, name, *control_regions_4j)

    name = "SR6JBVEM_meffInc30"
    assert name in signal_regions
    yield name, 3, region.prune(workspace, name, *control_regions_6j)


if __name__ == "__main__":
    main()
