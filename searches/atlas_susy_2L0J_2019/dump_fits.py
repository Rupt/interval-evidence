"""
time python searches/atlas_susy_2L0J_2019/dump_fits.py

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
        # DF 0J
        "SR_DF_0J_100_inf": (50, 250),
        "SR_DF_0J_160_inf": (10, 40),
        "SR_DF_0J_100_120": (20, 140),
        "SR_DF_0J_120_160": (20, 80),
        # DF 1J
        "SR_DF_1J_100_inf": (40, 140),
        "SR_DF_1J_160_inf": (5, 40),
        "SR_DF_1J_100_120": (10, 90),
        "SR_DF_1J_120_160": (10, 45),
        # SF 0J
        "SR_SF_0J_100_inf": (80, 260),
        "SR_SF_0J_160_inf": (25, 70),
        "SR_SF_0J_100_120": (15, 115),
        "SR_SF_0J_120_160": (30, 100),
        # SF 1J
        "SR_SF_1J_100_inf": (60, 260),
        "SR_SF_1J_160_inf": (20, 100),
        "SR_SF_1J_100_120": (0, 140),
        "SR_SF_1J_120_160": (25, 65),
    }

    region_name_to_anchors = {
        "SR_DF_0J_120_160": [60.0],
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump(name, lo, hi, region_name_to_anchors=region_name_to_anchors)


def dump(name, lo, hi, *, nbins=200, region_name_to_anchors=None):
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
