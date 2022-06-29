"""
time python searches/ins1767649/dump_fit_signal.py

"""

import os

from pyhf_stuff import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR_E_mll_1": (0, 15),
        "SR_E_mll_2": (0, 80),
        "SR_E_mll_3": (0, 100),
        "SR_E_mll_5": (0, 150),
        "SR_E_mll_10": (0, 200),
        "SR_E_mll_20": (0, 300),
        "SR_E_mll_30": (0, 300),
        "SR_E_mll_40": (0, 300),
        "SR_E_mll_60": (0, 300),
        "SR_S_100p5": (0, 60),
        "SR_S_101": (0, 50),
        "SR_S_102": (0, 100),
        "SR_S_105": (0, 150),
        "SR_S_110": (0, 150),
        "SR_S_120": (0, 200),
        "SR_S_130": (0, 200),
        "SR_S_140": (0, 250),
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
