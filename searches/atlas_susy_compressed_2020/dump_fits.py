"""
time python searches/atlas_susy_compressed_2020/dump_fits.py

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
        "SR_E_mll_1": (0, 6),
        "SR_E_mll_2": (10, 100),
        "SR_E_mll_3": (20, 150),
        "SR_E_mll_5": (80, 220),
        "SR_E_mll_10": (120, 300),
        "SR_E_mll_20": (200, 450),
        "SR_E_mll_30": (250, 550),
        "SR_E_mll_40": (300, 600),
        "SR_E_mll_60": (350, 700),
        "SR_S_100p5": (10, 60),
        "SR_S_101": (25, 200),
        "SR_S_102": (45, 200),
        "SR_S_105": (100, 260),
        "SR_S_110": (150, 400),
        "SR_S_120": (200, 460),
        "SR_S_130": (250, 600),
        "SR_S_140": (300, 650),
    }

    region_name_to_anchors = {}

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump(name, lo, hi, region_name_to_anchors=region_name_to_anchors)


def dump(name, lo, hi, *, nbins=50, region_name_to_anchors=None):
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
