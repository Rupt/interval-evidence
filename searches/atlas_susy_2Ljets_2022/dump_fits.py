"""
time python searches/atlas_susy_2Ljets_2022/dump_fits.py

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
        "ewk_high": (2, 8),
        "ewk_int": (20, 70),
        "ewk_llbb": (0, 3),
        "ewk_low": (0, 100),
        "ewk_offshell": (0, 40),
        "rjr_sr2l_isr": (0, 100),
        "rjr_sr2l_low": (0, 100),
        "str_src_12_31": (0, 20),
        "str_src_12_61": (0, 40),
        "str_src_31_81": (0, 40),
        "str_src_81": (0, 100),
        "str_srhigh_12_301": (0, 60),
        "str_srhigh_301": (0, 20),
        "str_srlow_101_201": (0, 80),
        "str_srlow_101_301": (0, 80),
        "str_srlow_12_81": (0, 35),
        "str_srlow_301": (0, 20),
        "str_srmed_101": (0, 100),
        "str_srmed_12_101": (20, 90),
        "str_srzhigh": (0, 20),
        "str_srzlow": (0, 60),
        "str_srzmed": (0, 40),
    }

    region_name_to_anchors = {
        "str_src_31_81": [36.0],
        "str_srhigh_301": [15.0],
        "str_src_81": [150.0],
        "str_srlow_101_301": [100.0],
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
