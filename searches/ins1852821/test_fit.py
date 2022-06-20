"""
time python searches/ins1852821/test_fit.py

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
        "SR0breq": (0, 5),
        "SR0bvetoloose": (5, 40),
        "SR0bvetotight": (0, 35),
        "SR0ZZbvetoloose": (0, 30),
        "SR0ZZbvetotight": (0, 5),
        "SR0ZZloose": (50, 550),
        "SR0ZZtight": (5, 45),
        "SR1breq": (0, 8),
        "SR1bvetoloose": (0, 20),
        "SR1bvetotight": (0, 8),
        "SR2breq": (0, 10),
        "SR2bvetoloose": (0, 50),
        "SR2bvetotight": (0, 10),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_cabinetry.fit(region_1).dump(dir_fit)
    fit_cabinetry_post.fit(region_1).dump(dir_fit)
    fit_normal.fit(region_1).dump(dir_fit)

    fit_linspace.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
