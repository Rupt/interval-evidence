"""
time python searches/atlas_susy_hb_2019/dump_fit_signal.py

"""

import os

from discohist import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SRA": (0, 40),
        "SRB": (0, 20),
        "SRC": (0, 100),
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
