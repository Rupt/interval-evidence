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

    sr_df_0j = ["SRDF_0%s_cuts" % c for c in "abcdefghi"]
    sr_df_1j = ["SRDF_1%s_cuts" % c for c in "abcdefghi"]
    sr_sf_0j = ["SRSF_0%s_cuts" % c for c in "abcdefghi"]
    sr_sf_1j = ["SRSF_1%s_cuts" % c for c in "abcdefghi"]
    assert set(sr_df_0j + sr_df_1j + sr_sf_0j + sr_sf_1j) == signal_regions

    sr_name_to_srs = {
        # DF 0J
        "SR_DF_0J_100_inf": sr_df_0j,
        "SR_DF_0J_160_inf": sr_df_0j[4:],
        "SR_DF_0J_100_120": sr_df_0j[:3],
        "SR_DF_0J_120_160": sr_df_0j[3:5],
        # DF 1J
        "SR_DF_1J_100_inf": sr_df_1j,
        "SR_DF_1J_160_inf": sr_df_1j[4:],
        "SR_DF_1J_100_120": sr_df_1j[:3],
        "SR_DF_1J_120_160": sr_df_1j[3:5],
        # SF 0J
        "SR_SF_0J_100_inf": sr_sf_0j,
        "SR_SF_0J_160_inf": sr_sf_0j[4:],
        "SR_SF_0J_100_120": sr_sf_0j[:3],
        "SR_SF_0J_120_160": sr_sf_0j[3:5],
        # SF 1J
        "SR_SF_1J_100_inf": sr_sf_1j,
        "SR_SF_1J_160_inf": sr_sf_1j[4:],
        "SR_SF_1J_100_120": sr_sf_1j[:3],
        "SR_SF_1J_120_160": sr_sf_1j[3:5],
    }

    for sr_name, srs in sr_name_to_srs.items():
        workspace_i = region.merge_channels(workspace, sr_name, srs)
        workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
        yield sr_name, workspace_i


if __name__ == "__main__":
    main()
