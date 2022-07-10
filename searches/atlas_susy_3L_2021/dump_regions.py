"""
Usage:

python searches/atlas_susy_3L_2021/dump_regions.py

"""
import os

import numpy
import pyhf

from discohisto import region, serial

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
    # combine sr fake stat modifiers for merging
    workspace = _combine_on_ffstat(workspace)

    def workspace_onshell(sr_name, indices):
        srs = ["SR%d_WZ_cuts" % i for i in indices]
        workspace_i = region.merge_channels(workspace, sr_name, srs)
        workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
        return sr_name, workspace_i

    yield workspace_onshell("incSR_WZ_1", [2, 3])
    yield workspace_onshell("incSR_WZ_2", [4])

    # 3: Table 16 says mT in [100, 160],  met in [150, 250], njets > 0
    # this leaves SR_WZ_10 with a relaxed HT cut
    # but SR_WZ_10 observes 11 data, but incSR_WZ_3 observes 4 :-(
    # yield workspace_onshell("incSR_WZ_3", [...])

    # 4, 5, 6 seem to use non-reproduceable cuts on HT and HTlep

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
    # 5/3?: data and backgrounds match 7 + 8
    # which would be mT > 160, met > 200 (labelled incSR_WZ_5 in Table 16)
    # choosing 3 (private communications) thank you, you know who you are
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

    # These parameters are either normsys or histosys in different SRs.
    # Arguably normsys would be the preferred target to avoid negative yields,
    # but in the case of multiple bins, histosys could not be converted there
    # since normsys scales all bins in a channel.
    # Combine them for merging.
    workspace = _convert_normsys_to_histosys(
        workspace,
        ["TL_ttbar_ISR", "TL_ttbar_FSR"],
    )

    # mcstat modifiers have SR-specific names; combine them for combinations
    workspace = _combine_shape_mcstat(workspace)

    # "Seven m^min_ll bins are defined with boundaries at
    # 1, 12, 15, 20, 30, 40, 60 and 75 GeV, and labelled ‘a’ to ‘g’"
    def workspace_offshell(sr_name, labels):
        srs = ["SR%s_cuts" % label for label in labels]
        workspace_i = region.merge_channels(workspace, sr_name, srs)
        workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
        return sr_name, workspace_i

    yield workspace_offshell("incSR_offWZ_highEt_nj_a", ["high_nJa"])
    yield workspace_offshell("incSR_offWZ_highEt_nj_b", ["high_nJb"])
    yield workspace_offshell(
        "incSR_offWZ_highEt_nj_c1", ["high_nJa", "high_nJb", "high_nJc"]
    )
    yield workspace_offshell("incSR_offWZ_highEt_nj_c2", ["high_nJc"])

    yield workspace_offshell("incSR_offWZ_lowEt_b", ["low_0Jb", "low_nJb"])
    yield workspace_offshell(
        "incSR_offWZ_lowEt_c", ["low_0Jb", "low_0Jc", "low_nJb", "low_nJc"]
    )
    yield workspace_offshell("incSR_offWZ_highEt_b", ["high_0Jb", "high_nJb"])
    yield workspace_offshell(
        "incSR_offWZ_highEt_c",
        ["high_0Jb", "high_0Jc", "high_nJb", "high_nJc"],
    )

    yield workspace_offshell(
        "incSR_offWZ_d",
        [
            "low_0Jb",
            "low_0Jc",
            "low_0Jd",
            "low_nJb",
            "low_nJc",
            "low_nJd",
            "high_0Jb",
            "high_0Jc",
            "high_0Jd",
            "high_nJb",
            "high_nJc",
            "high_nJd",
        ],
    )
    yield workspace_offshell(
        "incSR_offWZ_e1",
        [
            "low_0Jb",
            "low_0Jc",
            "low_0Jd",
            "low_0Je",
            "low_nJb",
            "low_nJc",
            "low_nJd",
            "low_nJe",
            "high_0Jb",
            "high_0Jc",
            "high_0Jd",
            "high_0Je",
            "high_nJb",
            "high_nJc",
            "high_nJd",
            "high_nJe",
        ],
    )
    yield workspace_offshell(
        "incSR_offWZ_e2",
        [
            "low_0Jc",
            "low_0Jd",
            "low_0Je",
            "low_nJc",
            "low_nJd",
            "low_nJe",
            "high_0Jc",
            "high_0Jd",
            "high_0Je",
            "high_nJc",
            "high_nJd",
            "high_nJe",
        ],
    )
    # Table 17 says mll in [12, 60], and c-f1. But c starts from mll=20
    # c-f1 yields 445 data
    # Table 19 shows 479 data
    # b-f1 yields 479 data
    # (I have emailed ATLAS publications about this)
    yield workspace_offshell(
        "incSR_offWZ_f1",
        [
            "low_0Jb",
            "low_0Jc",
            "low_0Jd",
            "low_0Je",
            "low_0Jf1",
            "low_0Jf2",
            "low_nJb",
            "low_nJc",
            "low_nJd",
            "low_nJe",
            "low_nJf1",
            "low_nJf2",
            "high_0Jb",
            "high_0Jc",
            "high_0Jd",
            "high_0Je",
            "high_0Jf1",
            "high_0Jf2",
            "high_nJb",
            "high_nJc",
            "high_nJd",
            "high_nJe",
            "high_nJf",
        ],
    )
    yield workspace_offshell(
        "incSR_offWZ_f2",
        [
            "low_0Je",
            "low_0Jf1",
            "low_0Jf2",
            "low_nJe",
            "low_nJf1",
            "low_nJf2",
            "high_0Je",
            "high_0Jf1",
            "high_0Jf2",
            "high_nJe",
            "high_nJf",
        ],
    )
    yield workspace_offshell(
        "incSR_offWZ_g1",
        [
            "low_0Jb",
            "low_0Jc",
            "low_0Jd",
            "low_0Je",
            "low_0Jf1",
            "low_0Jf2",
            "low_0Jg1",
            "low_0Jg2",
            "low_nJb",
            "low_nJc",
            "low_nJd",
            "low_nJe",
            "low_nJf1",
            "low_nJf2",
            "low_nJg1",
            "low_nJg2",
            "high_0Jb",
            "high_0Jc",
            "high_0Jd",
            "high_0Je",
            "high_0Jf1",
            "high_0Jf2",
            "high_0Jg1",
            "high_0Jg2",
            "high_nJb",
            "high_nJc",
            "high_nJd",
            "high_nJe",
            "high_nJf",
            "high_nJg",
        ],
    )
    yield workspace_offshell(
        "incSR_offWZ_g2",
        [
            "low_0Je",
            "low_0Jf1",
            "low_0Jf2",
            "low_0Jg1",
            "low_0Jg2",
            "low_nJe",
            "low_nJf1",
            "low_nJf2",
            "low_nJg1",
            "low_nJg2",
            "high_0Je",
            "high_0Jf1",
            "high_0Jf2",
            "high_0Jg1",
            "high_0Jg2",
            "high_nJe",
            "high_nJf",
            "high_nJg",
        ],
    )
    yield workspace_offshell(
        "incSR_offWZ_g3",
        [
            "low_0Jf1",
            "low_0Jf2",
            "low_0Jg1",
            "low_0Jg2",
            "low_nJf1",
            "low_nJf2",
            "low_nJg1",
            "low_nJg2",
            "high_0Jf1",
            "high_0Jf2",
            "high_0Jg1",
            "high_0Jg2",
            "high_nJf",
            "high_nJg",
        ],
    )
    yield workspace_offshell(
        "incSR_offWZ_g4",
        [
            "low_0Jg1",
            "low_0Jg2",
            "low_nJg1",
            "low_nJg2",
            "high_0Jg1",
            "high_0Jg2",
            "high_nJg",
        ],
    )

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


def _convert_normsys_to_histosys(workspace, modifier_names):
    modifier_names = set(modifier_names)

    channels_new = []
    for channel in workspace["channels"]:
        samples_new = []
        for sample in channel["samples"]:
            sample_data = numpy.array(sample["data"])
            modifiers_new = []
            for modifier in sample["modifiers"]:
                name = modifier["name"]
                if not (
                    name in modifier_names and modifier["type"] == "normsys"
                ):
                    modifiers_new.append(modifier)
                    continue

                data = modifier["data"]
                hi = data["hi"]
                lo = data["lo"]
                data_new = {
                    "hi_data": (hi * sample_data).tolist(),
                    "lo_data": (lo * sample_data).tolist(),
                }
                modifiers_new.append(
                    {
                        "data": data_new,
                        "name": name,
                        "type": "histosys",
                    }
                )
            samples_new.append(dict(sample, modifiers=modifiers_new))
        channels_new.append(dict(channel, samples=samples_new))

    newspec = {
        "channels": channels_new,
        "measurements": workspace["measurements"],
        "observations": workspace["observations"],
        "version": workspace["version"],
    }
    return pyhf.Workspace(newspec)


def _combine_on_ffstat(workspace):
    modifider_renames = {}

    for channel in workspace["channels"]:
        if not channel["name"].startswith("SR"):
            continue
        for sample in channel["samples"]:
            for modifier in sample["modifiers"]:
                name = modifier["name"]
                if name.startswith("ON_FFstat_"):
                    modifider_renames[name] = "ON_FFstat_SR_rename"

    return workspace.rename(modifiers=modifider_renames)


def _combine_shape_mcstat(workspace):
    modifider_renames = {}

    for channel in workspace["channels"]:
        if not channel["name"].startswith("SR"):
            continue
        for sample in channel["samples"]:
            for modifier in sample["modifiers"]:
                name = modifier["name"]
                if name.startswith("shape_mcstat_Fakes_"):
                    modifider_renames[name] = "shape_mcstat_Fakes_SR_rename"
                    continue
                if name.startswith("shape_mcstat_Others_"):
                    modifider_renames[name] = "shape_mcstat_Others_SR_rename"
                    continue
                if name.startswith("shape_mcstat_ttbar_"):
                    modifider_renames[name] = "shape_mcstat_ttbar_SR_rename"
                    continue
                if name.startswith("shape_mcstat_ttX_"):
                    modifider_renames[name] = "shape_mcstat_ttX_SR_rename"
                    continue
                if name.startswith("shape_mcstat_WZ0j_"):
                    modifider_renames[name] = "shape_mcstat_WZ0j_SR_rename"
                    continue
                if name.startswith("shape_mcstat_WZnj_"):
                    modifider_renames[name] = "shape_mcstat_WZnj_SR_rename"
                    continue
                if name.startswith("shape_mcstat_ZZ_"):
                    modifider_renames[name] = "shape_mcstat_ZZ_SR_rename"
                    continue

    return workspace.rename(modifiers=modifider_renames)


if __name__ == "__main__":
    main()
