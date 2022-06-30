"""
time python searches/ins1866951/dump_fit_signal.py

"""

import os

from pyhf_stuff import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "incSR_WZ_1": (0, 50),
        "incSR_WZ_2": (0, 20),
        "incSR_WZ_3": (0, 30),
        "incSR_offWZ_highEt_nj_a": (0, 20),
        "incSR_offWZ_highEt_nj_b": (0, 20),
        "incSR_offWZ_highEt_nj_c1": (0, 30),
        "incSR_offWZ_highEt_nj_c2": (0, 20),
        "incSR_offWZ_lowEt_b": (0, 50),
        "incSR_offWZ_highEt_b": (0, 20),
        "incSR_offWZ_lowEt_c": (0, 100),
        "incSR_offWZ_highEt_c": (0, 40),
        "incSR_offWZ_d": (0, 150),
        "incSR_offWZ_e1": (0, 200),
        "incSR_offWZ_e2": (0, 200),
        "incSR_offWZ_f1": (0, 200),
        "incSR_offWZ_f2": (0, 150),
        "incSR_offWZ_g1": (0, 250),
        "incSR_offWZ_g2": (0, 200),
        "incSR_offWZ_g3": (0, 150),
        "incSR_offWZ_g4": (0, 120),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=200):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_signal.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
