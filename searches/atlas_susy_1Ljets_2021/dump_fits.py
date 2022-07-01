"""
time python searches/atlas_susy_1Ljets_2021/dump_fits.py

"""

import os

from discohist import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_linspace,
    fit_normal,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR2JBVEM_meffInc30_gluino": (5, 40),
        "SR2JBVEM_meffInc30_squark": (40, 160),
        "SR4JhighxBVEM_meffInc30": (0, 30),
        "SR4JlowxBVEM_meffInc30": (0, 30),
        "SR6JBVEM_meffInc30_gluino": (0, 12),
        "SR6JBVEM_meffInc30_squark": (0, 12),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=200):
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
