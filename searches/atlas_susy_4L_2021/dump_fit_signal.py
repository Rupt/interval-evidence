"""
time python searches/atlas_susy_4L_2021/dump_fit_signal.py

"""

import os

from discohisto import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR0breq": (0, 20),
        "SR0bvetoloose": (0, 40),
        "SR0bvetotight": (0, 30),
        "SR0ZZbvetoloose": (0, 30),
        "SR0ZZbvetotight": (0, 30),
        "SR0ZZloose": (0, 250),
        "SR0ZZtight": (0, 40),
        "SR1breq": (0, 20),
        "SR1bvetoloose": (0, 30),
        "SR1bvetotight": (0, 20),
        "SR2breq": (0, 20),
        "SR2bvetoloose": (0, 35),
        "SR2bvetotight": (0, 20),
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
