"""
time python searches/ins1852821/test_fit_signal.py

"""

import os

from pyhf_stuff import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR0breq": (0, 10),
        "SR0bvetoloose": (0, 20),
        "SR0bvetotight": (0, 10),
        "SR0ZZbvetoloose": (0, 10),
        "SR0ZZbvetotight": (0, 20),
        "SR0ZZloose": (0, 100),
        "SR0ZZtight": (0, 20),
        "SR1breq": (0, 10),
        "SR1bvetoloose": (0, 10),
        "SR1bvetotight": (0, 10),
        "SR2breq": (0, 10),
        "SR2bvetoloose": (0, 20),
        "SR2bvetotight": (0, 10),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_signal.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
