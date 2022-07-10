"""
time python searches/atlas_susy_2L0J_2019/dump_fit_signal.py

"""

import os

from discohist import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        # DF 0J
        "SR_DF_0J_100_inf": (0, 80),
        "SR_DF_0J_160_inf": (0, 40),
        "SR_DF_0J_100_120": (0, 50),
        "SR_DF_0J_120_160": (0, 50),
        # DF 1J
        "SR_DF_1J_100_inf": (0, 80),
        "SR_DF_1J_160_inf": (0, 40),
        "SR_DF_1J_100_120": (0, 50),
        "SR_DF_1J_120_160": (0, 50),
        # SF 0J
        "SR_SF_0J_100_inf": (0, 100),
        "SR_SF_0J_160_inf": (0, 40),
        "SR_SF_0J_100_120": (0, 80),
        "SR_SF_0J_120_160": (0, 50),
        # SF 1J
        "SR_SF_1J_100_inf": (0, 100),
        "SR_SF_1J_160_inf": (0, 40),
        "SR_SF_1J_100_120": (0, 100),
        "SR_SF_1J_120_160": (0, 50),
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
