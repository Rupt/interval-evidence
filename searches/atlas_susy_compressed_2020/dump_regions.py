"""
Usage:

python searches/atlas_susy_compressed_2020/dump_regions.py

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
    # ewkinos: SR-E
    spec = serial.load_json_gz(os.path.join(BASEPATH, "ewkinos_bkg.json.gz"))
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

    # after badly approximating results in the paper, I've had a go at fixing
    # quirks of the merged likelihoods. These changes make very little
    # difference to the result, but I think they make senses
    # some normsys / histosys clashes
    bad_mods = _get_bad_repeated_mods(workspace)

    def bad(modifier, sample, channel):
        key = (channel["name"], sample["name"], modifier["name"])
        return key in bad_mods

    workspace = region.filter_modifiers(workspace, [bad])
    # repeated normfactors both multiplying the SR
    workspace = _merge_normfactors(workspace)
    # separate parameters for all the fakes
    workspace = _merge_fake_stuff(workspace, "disco")
    # maybe they merged the CRs? also doesn't explain it
    if MERGE_CRS:
        workspace = _merge_ewkino_crs(workspace)
        control_regions = {"CRVV_MLL", "CRtau_MLL", "CRtop_MLL"}

    # SR-E-high
    sr_e_high_ee = "SRee_eMLL%s_hghmet_cuts"  # c..h
    sr_e_high_mm = "SRmm_eMLL%s_hghmet_cuts"  # a..h

    # SR-E-med
    sr_e_med_ee = "SRee_eMLL%s_lowmet_deltaM_low_cuts"  # c..h
    sr_e_med_mm = "SRmm_eMLL%s_lowmet_deltaM_low_cuts"  # a..h

    # yes, deltaM_high corresponds to SR-E-low
    # and deltaM_low corresponds to SR-E-med
    # SR-E-low
    sr_e_low_ee = "SRee_eMLL%s_lowmet_deltaM_high_cuts"  # c..h
    sr_e_low_mm = "SRmm_eMLL%s_lowmet_deltaM_high_cuts"  # a..h

    # SR-E-1l1t
    sr_e_l1lt = "SR_eMLL%s_Onelep1track_cuts"  # a..f

    # https://doi.org/10.1103/PhysRevD.101.052005
    # DR boundaries are < [1, 2, 3, 5, 10, 20, 30, 40, 60]
    # selecting bins from Tables XI and XII
    # 1
    sr_name = "SR_E_mll_1"
    srs = [
        *(sr_e_l1lt % c for c in "a"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 2
    sr_name = "SR_E_mll_2"
    srs = [
        *(sr_e_high_mm % c for c in "a"),
        *(sr_e_med_mm % c for c in "a"),
        *(sr_e_low_mm % c for c in "a"),
        *(sr_e_l1lt % c for c in "abc"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 3
    sr_name = "SR_E_mll_3"
    srs = [
        *(sr_e_high_mm % c for c in "ab"),
        *(sr_e_med_mm % c for c in "ab"),
        *(sr_e_low_mm % c for c in "ab"),
        *(sr_e_l1lt % c for c in "abcd"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 5
    sr_name = "SR_E_mll_5"
    srs = [
        *(sr_e_high_ee % c for c in "c"),
        *(sr_e_high_mm % c for c in "abc"),
        *(sr_e_med_ee % c for c in "c"),
        *(sr_e_med_mm % c for c in "abc"),
        *(sr_e_low_ee % c for c in "c"),
        *(sr_e_low_mm % c for c in "abc"),
        *(sr_e_l1lt % c for c in "abcdef"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 10
    sr_name = "SR_E_mll_10"
    srs = [
        *(sr_e_high_ee % c for c in "cd"),
        *(sr_e_high_mm % c for c in "abcd"),
        *(sr_e_med_ee % c for c in "cd"),
        *(sr_e_med_mm % c for c in "abcd"),
        *(sr_e_low_ee % c for c in "cd"),
        *(sr_e_low_mm % c for c in "abcd"),
        *(sr_e_l1lt % c for c in "abcdef"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 20
    sr_name = "SR_E_mll_20"
    srs = [
        *(sr_e_high_ee % c for c in "cde"),
        *(sr_e_high_mm % c for c in "abcde"),
        *(sr_e_med_ee % c for c in "cde"),
        *(sr_e_med_mm % c for c in "abcde"),
        *(sr_e_low_ee % c for c in "cde"),
        *(sr_e_low_mm % c for c in "abcde"),
        *(sr_e_l1lt % c for c in "abcdef"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 30
    sr_name = "SR_E_mll_30"
    srs = [
        *(sr_e_high_ee % c for c in "cdef"),
        *(sr_e_high_mm % c for c in "abcdef"),
        *(sr_e_med_ee % c for c in "cdef"),
        *(sr_e_med_mm % c for c in "abcdef"),
        *(sr_e_low_ee % c for c in "cdef"),
        *(sr_e_low_mm % c for c in "abcdef"),
        *(sr_e_l1lt % c for c in "abcdef"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 40
    sr_name = "SR_E_mll_40"
    srs = [
        *(sr_e_high_ee % c for c in "cdefg"),
        *(sr_e_high_mm % c for c in "abcdefg"),
        *(sr_e_med_ee % c for c in "cdefg"),
        *(sr_e_med_mm % c for c in "abcdefg"),
        *(sr_e_low_ee % c for c in "cdefg"),
        *(sr_e_low_mm % c for c in "abcdefg"),
        *(sr_e_l1lt % c for c in "abcdef"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # 60
    sr_name = "SR_E_mll_60"
    srs = [
        *(sr_e_high_ee % c for c in "cdefgh"),
        *(sr_e_high_mm % c for c in "abcdefgh"),
        *(sr_e_med_ee % c for c in "cdefgh"),
        *(sr_e_med_mm % c for c in "abcdefgh"),
        *(sr_e_low_ee % c for c in "cdefgh"),
        *(sr_e_low_mm % c for c in "abcdefgh"),
        *(sr_e_l1lt % c for c in "abcdef"),
    ]
    workspace_i = region.merge_channels(workspace, sr_name, srs)
    workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
    yield sr_name, workspace_i

    # sleptons: SR-S
    spec = serial.load_json_gz(os.path.join(BASEPATH, "sleptons_bkg.json.gz"))
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

    # some normsys / histosys clashes
    bad_mods = _get_bad_repeated_mods(workspace)

    def bad(modifier, sample, channel):
        key = (channel["name"], sample["name"], modifier["name"])
        return key in bad_mods

    workspace = region.filter_modifiers(workspace, [bad])
    workspace = _merge_normfactors(workspace)
    workspace = _merge_fake_stuff(workspace, "disco")
    if MERGE_CRS:
        workspace = _merge_slepton_crs(workspace)
        control_regions = {"CRVV_MT2", "CRtau_MT2", "CRtop_MT2"}

    # SR-S-high
    sr_s_high_ee = "SRee_eMT2%s_hghmet_cuts"  # a..h
    sr_s_high_mm = "SRmm_eMT2%s_hghmet_cuts"  # a..h

    # SR-S-low
    sr_s_low_ee = "SRee_eMT2%s_lowmet_V2_cuts"  # a..h
    sr_s_low_mm = "SRmm_eMT2%s_lowmet_V2_cuts"  # a..h

    # mt2 boundaries at 100, 100.5, 101, 102, 105, 110, 120, 130, and 140 GeV
    # reading indices from Table XIV
    # SR-S combines high and low

    sr_name_to_alphabet = {
        "SR_S_100p5": "a",
        "SR_S_101": "ab",
        "SR_S_102": "abc",
        "SR_S_105": "abcd",
        "SR_S_110": "abcde",
        "SR_S_120": "abcdef",
        "SR_S_130": "abcdefg",
        "SR_S_140": "abcdefgh",
    }

    for sr_name, alphabet in sr_name_to_alphabet.items():
        srs = [
            *(sr_s_high_ee % c for c in alphabet),
            *(sr_s_high_mm % c for c in alphabet),
            *(sr_s_low_ee % c for c in alphabet),
            *(sr_s_low_mm % c for c in alphabet),
        ]
        workspace_i = region.merge_channels(workspace, sr_name, srs)
        workspace_i = region.prune(workspace_i, [sr_name, *control_regions])
        yield sr_name, workspace_i

    # unfortunately, SR-VBF is not included in HEPData


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


def _merge_normfactors(workspace):
    # there are separate normfactors for low and high met
    # we don't want their product in the signal region, so combine them
    workspace = region.merge_normfactor(
        workspace, "mu_Ztt", ["mu_Ztt_hghmet", "mu_Ztt_lowmet"]
    )
    workspace = region.merge_normfactor(
        workspace, "mu_VV", ["mu_VV_hghmet", "mu_VV_lowmet"]
    )
    workspace = region.merge_normfactor(
        workspace, "mu_top", ["mu_top_hghmet", "mu_top_lowmet"]
    )
    return workspace


def _merge_fake_stuff(workspace, name):
    # find all modifiers
    modifiers_in_srs = set()

    for channel in workspace["channels"]:
        if not channel["name"].startswith("SR"):
            continue
        for sample in channel["samples"]:
            for modifier in sample["modifiers"]:
                modifiers_in_srs.add(modifier["name"])

    # construct the rename map
    rename_modifiers = {}
    name_ff = "FF_syst_stat_rename_" + name
    name_shape = "shape_fakes_stat_fakes_rename_" + name
    name_zero = "fake_zeroEstimate_UL_rename_" + name

    for modifier_name in modifiers_in_srs:
        if modifier_name.startswith("FF_syst_stat_"):
            rename_modifiers[modifier_name] = name_ff
            continue
        if modifier_name.startswith("shape_fakes_stat_fakes_"):
            rename_modifiers[modifier_name] = name_shape
            continue
        if modifier_name.startswith("fake_zeroEstimate_UL_"):
            rename_modifiers[modifier_name] = name_zero
            continue

    return workspace.rename(modifiers=rename_modifiers)


def _merge_ewkino_crs(workspace):
    workspace = region.merge_channels(
        workspace, "CRVV_MLL", ["CRVV_MLL_hghmet_cuts", "CRVV_MLL_lowmet_cuts"]
    )
    workspace = region.merge_channels(
        workspace,
        "CRtau_MLL",
        ["CRtau_MLL_hghmet_cuts", "CRtau_MLL_lowmet_cuts"],
    )
    workspace = region.merge_channels(
        workspace,
        "CRtop_MLL",
        ["CRtop_MLL_hghmet_cuts", "CRtop_MLL_lowmet_cuts"],
    )
    return workspace


def _merge_slepton_crs(workspace):
    workspace = region.merge_channels(
        workspace, "CRVV_MT2", ["CRVV_MT2_hghmet_cuts", "CRVV_MT2_lowmet_cuts"]
    )
    workspace = region.merge_channels(
        workspace,
        "CRtau_MT2",
        ["CRtau_MT2_hghmet_cuts", "CRtau_MT2_lowmet_cuts"],
    )
    workspace = region.merge_channels(
        workspace,
        "CRtop_MT2",
        ["CRtop_MT2_hghmet_cuts", "CRtop_MT2_lowmet_cuts"],
    )
    return workspace


if __name__ == "__main__":
    main()
