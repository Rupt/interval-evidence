"""
Usage:

python searches/ins1866951/dump_regions.py

"""
import os

import pyhf

from pyhf_stuff import region, serial

BASEPATH = os.path.dirname(__file__)

# an experiment which didn't reproduce paper results
MERGE_CRS = False


def main():
    for sr_name, workspace in generate_regions():
        region.Region(sr_name, workspace).dump(
            os.path.join(BASEPATH, sr_name),
        )


def generate_regions():
    # onshell: WZ
    spec = serial.load_json_gz(os.path.join(BASEPATH, "onshell_bkg.json.gz"))
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

    # workspace cleanup: some normsys / histosys clashes
    bad_mods = _get_bad_repeated_mods(workspace)

    def bad(modifier, sample, channel):
        key = (channel["name"], sample["name"], modifier["name"])
        return key in bad_mods

    workspace = region.filter_modifiers(workspace, [bad])

    def workspace_onshell(sr_name, indices):
        srs = ["SR%d_WZ_cuts" % i for i in indices]
        workspace_i = region.merge_channels(workspace, sr_name, srs)
        workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
        return sr_name, workspace_i

    yield workspace_onshell("incSR_WZ_1", [2, 3])
    yield workspace_onshell("incSR_WZ_2", [4])

    # 3: Table 16 says mT in [100, 160],  met in [150, 250], njets > 0
    # this leaves SR_WZ_10 with a relaxed HT cut
    # but SR_WZ_10 observed 11 data, incSR_WZ_3 observes 4
    # yield workspace_onshell("incSR_WZ_3", [...])

    # 4: Table 16 says mT in [100, 160], met > 250, njets > 0
    # this leaves SR_WZ_11 (which observes 0) and SR_WZ_12 (which observes 0)
    # but incSR_WZ_4 observes 34 data
    # yield workspace_onshell("incSR_WZ_4", [11, 12])

    # 5: can only be 7 + 8, which observe 3, 1
    # but incSR_WZ_5 observes 1 (expected 5.2 though - maybe error?)
    # yield workspace_onshell("incSR_WZ_5", [7, 8])

    # 6: no obvious candidates
    # incSR_WZ_6 observes 23
    # can't be 18 + 19 + 20 (9, 3, 1)
    # can't be 14, 15, 16 (no boundary at 200)

    # Redux! Distrust the tables, try to reverse engineer from results.
    # 3: data and backgrounds match 7 + 8
    # which would be mT > 160, met > 200 (labelled incSR_WZ_5 in Table 16)
    yield workspace_onshell("incSR_WZ_3", [7, 8])

    # offshell: offWZ x low/high x 0j/nj
    spec = serial.load_json_gz(os.path.join(BASEPATH, "offshell_bkg.json.gz"))
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

    # "Seven m^min_ll bins are defined with boundaries at
    # 1, 12, 15, 20, 30, 40, 60 and 75 GeV, and labelled ‘a’ to ‘g’"
    def workspace_offshell(sr_name, labels):
        srs = ["SR%s_cuts" % label for label in labels]
        workspace_i = region.merge_channels(workspace, sr_name, srs)
        workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
        return sr_name, workspace_i

    yield workspace_offshell("incSR_offWZ_highEt_nja", ["high_nJa"])
    yield workspace_offshell("incSR_offWZ_highEt_njb", ["high_nJb"])
    yield workspace_offshell(
        "incSR_offWZ_highEt_njc1", ["high_nJa", "high_nJb", "high_nJc"]
    )
    yield workspace_offshell("incSR_offWZ_highEt_njc2", ["high_nJc"])
    # TODO more

    # unfortunately, SR-Wh are not included in HEPData
    # nor are SR3l-low and SR3l-ISR


def _get_bad_repeated_mods(workspace):
    """Return bad (channel, sample, modifier) trios

    These modifiers appear as both histosys and normsys, and the histosys has
    equal up and down which match the sample data.
    """
    bad_mods = set()

    for channel in workspace["channels"]:
        for sample in channel["samples"]:
            data = sample["data"]

            # identify normsys
            mods_normsys = set()
            for modifier in sample["modifiers"]:
                if modifier["type"] != "normsys":
                    continue
                mods_normsys.add(modifier["name"])

            # check against histosys
            for modifier in sample["modifiers"]:
                if modifier["type"] != "histosys":
                    continue
                if modifier["name"] not in mods_normsys:
                    continue

                hi = modifier["data"]["hi_data"]
                lo = modifier["data"]["lo_data"]
                # I don't know what to do if this fails
                assert hi == lo == data
                # but it seems to always succeed :-)

                key = (channel["name"], sample["name"], modifier["name"])
                bad_mods.add(key)

    return bad_mods


if __name__ == "__main__":
    main()
