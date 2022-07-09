"""
time python searches/atlas_susy_3Lresonance_2020/dump_fit_signal.py

"""

import os

from discohist import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        # SRFR
        "SRFR_90_110_all": (0, 15),
        "SRFR_110_130_all": (0, 20),
        "SRFR_150_170_all": (0, 30),
        "SRFR_170_190_all": (0, 20),
        "SRFR_190_210_all": (0, 20),
        "SRFR_210_230_all": (0, 20),
        "SRFR_230_250_all": (0, 20),
        "SRFR_250_270_all": (0, 20),
        "SRFR_270_300_all": (0, 20),
        "SRFR_300_330_all": (0, 15),
        "SRFR_330_360_all": (0, 15),
        "SRFR_360_400_all": (0, 20),
        "SRFR_400_440_all": (0, 20),
        "SRFR_440_580_all": (0, 12),
        "SRFR_580_inf_all": (0, 12),
        # SR4L
        "SR4l_90_110_all": (0, 20),
        "SR4l_110_130_all": (0, 30),
        "SR4l_150_170_all": (0, 22),
        "SR4l_170_190_all": (0, 25),
        "SR4l_190_210_all": (0, 20),
        "SR4l_210_230_all": (0, 20),
        "SR4l_230_250_all": (0, 20),
        "SR4l_250_270_all": (0, 20),
        "SR4l_270_300_all": (0, 20),
        "SR4l_300_330_all": (0, 20),
        "SR4l_330_360_all": (0, 20),
        "SR4l_360_400_all": (0, 20),
        "SR4l_400_440_all": (0, 20),
        "SR4l_440_580_all": (0, 20),
        "SR4l_580_inf_all": (0, 20),
        # SR3L
        "SR3l_90_110_all": (0, 20),
        "SR3l_110_130_all": (0, 20),
        "SR3l_150_170_all": (0, 20),
        "SR3l_170_190_all": (0, 20),
        "SR3l_190_210_all": (0, 20),
        "SR3l_210_230_all": (0, 20),
        "SR3l_230_250_all": (0, 20),
        "SR3l_250_270_all": (0, 15),
        "SR3l_270_300_all": (0, 15),
        "SR3l_300_330_all": (0, 15),
        "SR3l_330_360_all": (0, 15),
        "SR3l_360_400_all": (0, 15),
        "SR3l_400_440_all": (0, 20),
        "SR3l_440_580_all": (0, 20),
        "SR3l_580_inf_all": (0, 20),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_signal(name, lo, hi)


def dump_signal(name, lo, hi, nbins=200):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_signal.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
