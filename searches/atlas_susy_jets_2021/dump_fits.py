"""
time python searches/atlas_susy_jets_2021/dump_fits.py

"""

import os

from discohisto import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_linspace,
    fit_normal,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "BDT-GGd1": (10, 60),
        "BDT-GGd2": (30, 80),
        "BDT-GGd3": (150, 350),
        "BDT-GGd4": (200, 450),
        "BDT-GGo1": (0, 20),
        "BDT-GGo2": (5, 35),
        "BDT-GGo3": (50, 130),
        "BDT-GGo4": (100, 300),
        "SR2j-1600": (1500, 3000),
        "SR2j-2200": (750, 1250),
        "SR2j-2800": (50, 125),
        "SR4j-1000": (400, 700),
        "SR4j-2200": (40, 85),
        "SR4j-3400": (2, 12),
        "SR5j-1600": (250, 450),
        "SR6j-1000": (5, 40),
        "SR6j-2200": (0.1, 30),
        "SR6j-3400": (0, 10),
    }

    region_name_to_anchors = {
        "BDT-GGd2": [32.7],
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_fits(name, lo, hi, region_name_to_anchors=region_name_to_anchors)


def dump_fits(name, lo, hi, *, nbins=200, region_name_to_anchors=None):
    if region_name_to_anchors is None:
        region_name_to_anchors = {}

    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    # cabinetry
    fit_cabinetry.fit(region_1).dump(dir_fit)
    fit_cabinetry_post.fit(region_1).dump(dir_fit)

    # normal
    fit_normal.fit(region_1).dump(dir_fit)

    # linspace
    fit_linspace.fit(
        region_1,
        lo,
        hi,
        nbins + 1,
        anchors=region_name_to_anchors.get(name),
    ).dump(dir_fit)


if __name__ == "__main__":
    main()
