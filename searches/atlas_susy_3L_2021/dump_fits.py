"""
time python searches/atlas_susy_3L_2021/dump_fits.py

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
        "incSR_WZ_1": (10, 70),
        "incSR_WZ_2": (0, 5),
        "incSR_WZ_3": (0, 15),
        "incSR_offWZ_highEt_nj_a": (0, 18),
        "incSR_offWZ_highEt_nj_b": (0, 8),
        "incSR_offWZ_highEt_nj_c1": (0, 25),
        "incSR_offWZ_highEt_nj_c2": (0, 10),
        "incSR_offWZ_lowEt_b": (20, 60),
        "incSR_offWZ_highEt_b": (0, 30),
        "incSR_offWZ_lowEt_c": (50, 125),
        "incSR_offWZ_highEt_c": (0, 20),
        "incSR_offWZ_d": (125, 250),
        "incSR_offWZ_e1": (200, 400),
        "incSR_offWZ_e2": (200, 350),
        "incSR_offWZ_f1": (350, 600),
        "incSR_offWZ_f2": (200, 350),
        "incSR_offWZ_g1": (450, 750),
        "incSR_offWZ_g2": (300, 500),
        "incSR_offWZ_g3": (200, 400),
        "incSR_offWZ_g4": (80, 200),
    }

    region_name_to_anchors = {
        "incSR_offWZ_highEt_b": [15.0],
    }

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
