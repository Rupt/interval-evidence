"""
time python searches/ins1831992/dump_fits.py

"""

import os

from pyhf_stuff import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_linspace,
    fit_normal,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        # SRFR
        "SRFR_90_110_all": (0, 5),
        "SRFR_110_130_all": (0, 15),
        "SRFR_150_170_all": (0, 15),
        "SRFR_170_190_all": (0, 10),
        "SRFR_190_210_all": (0, 10),
        "SRFR_210_230_all": (0, 50),
        "SRFR_230_250_all": (0, 5),
        "SRFR_250_270_all": (0, 5),
        "SRFR_270_300_all": (0, 8),
        "SRFR_300_330_all": (0, 3),
        "SRFR_330_360_all": (0, 4),
        "SRFR_360_400_all": (0, 8),
        "SRFR_400_440_all": (0, 2),
        "SRFR_440_580_all": (0, 8),
        "SRFR_580_inf_all": (0, 10),
        # SR4L
        "SR4l_90_110_all": (0, 13),
        "SR4l_110_130_all": (10, 25),
        "SR4l_150_170_all": (5, 15),
        "SR4l_170_190_all": (0, 12),
        "SR4l_190_210_all": (0, 12),
        "SR4l_210_230_all": (0, 6),
        "SR4l_230_250_all": (0, 6),
        "SR4l_250_270_all": (0, 8),
        "SR4l_270_300_all": (0, 4),
        "SR4l_300_330_all": (0, 3),
        "SR4l_330_360_all": (0, 8),
        "SR4l_360_400_all": (0, 8),
        "SR4l_400_440_all": (0, 8),
        "SR4l_440_580_all": (0, 8),
        "SR4l_580_inf_all": (0, 10),
        # SR3L
        "SR3l_90_110_all": (0, 3),
        "SR3l_110_130_all": (0, 6),
        "SR3l_150_170_all": (0, 10),
        "SR3l_170_190_all": (0, 7),
        "SR3l_190_210_all": (0, 7),
        "SR3l_210_230_all": (0, 10),
        "SR3l_230_250_all": (0, 10),
        "SR3l_250_270_all": (0, 5),
        "SR3l_270_300_all": (0, 7),
        "SR3l_300_330_all": (0, 7),
        "SR3l_330_360_all": (0, 8),
        "SR3l_360_400_all": (0, 8),
        "SR3l_400_440_all": (0, 4),
        "SR3l_440_580_all": (0, 8),
        "SR3l_580_inf_all": (0, 8),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    # cabinetry fits fail here by default
    # Diboson3L in CRWZ and CRttZ shares a number of 3L theory normfactors
    # removing this of these appears to resolve the problem
    # (muRmuF or mu_Diboson3l could also be removed with similar results)
    def cr_theory(modifier, sample, channel):
        return (
            channel["name"] == "CRWZ_all_cuts"
            and modifier["name"] == "theory_scale_muR_Diboson3l"
        )

    region_cabinetry = region.Region(
        region_1.signal_region_name,
        region.filter_modifiers(region_1.workspace, [cr_theory]),
    )

    fit_cabinetry.fit(region_cabinetry).dump(dir_fit)
    fit_cabinetry_post.fit(region_cabinetry).dump(dir_fit)

    # normal
    fit_normal.fit(region_1).dump(dir_fit)

    # linspace
    fit_linspace.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
