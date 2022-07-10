"""
Usage:

python searches/atlas_susy_1Lbb_2020/dump_regions.py

"""
import os

import pyhf

from discohisto import region, serial

BASEPATH = os.path.dirname(__file__)


def main():
    for name, sr_name, workspace in generate_regions():
        region.Region(sr_name, workspace).dump(
            os.path.join(BASEPATH, name),
        )


def generate_regions():
    spec = serial.load_json_gz(os.path.join(BASEPATH, "bkg.json.gz"))
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

    # Table 2 of https://doi.org/10.1140/epjc/s10052-020-8050-3 says
    # mCT [GeV] (disc.) > 180
    # but the text below says
    # "For model-independent limits and null-hypothesis tests (‘disc.’ for
    # discovery), the various mCT bins are merged for each of the three SRs."
    # so we merge those three bins
    bins = range(3)

    # {'SRLMEM_mct2', 'SRHMEM_mct2', 'SRMMEM_mct2'}
    name = "SR_LM_disc"
    sr_name = "SRLMEM_mct2"
    workspace_i = region.prune(workspace, [sr_name, *control_regions])
    workspace_i = region.merge_bins(workspace_i, sr_name, bins)
    yield name, sr_name, workspace_i

    name = "SR_MM_disc"
    sr_name = "SRMMEM_mct2"
    workspace_i = region.prune(workspace, [sr_name, *control_regions])
    workspace_i = region.merge_bins(workspace_i, sr_name, bins)
    yield name, sr_name, workspace_i

    name = "SR_HM_disc"
    sr_name = "SRHMEM_mct2"
    workspace_i = region.prune(workspace, [sr_name, *control_regions])
    workspace_i = region.merge_bins(workspace_i, sr_name, bins)
    yield name, sr_name, workspace_i


if __name__ == "__main__":
    main()
