"""
time python searches/ins1788448/dump_fits.py

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
        "SRMU": (0, 4),
        "SRMET": (0, 2.5),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    # cabinetry
    fit_cabinetry.fit(region_1).dump(dir_fit)
    fit_cabinetry_post.fit(region_1).dump(dir_fit)

    # normal
    fit_normal.fit(region_1).dump(dir_fit)

    # linspace
    fit_linspace.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
