"""
time python searches/ins1755298/dump_fits.py

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
        "SR_LM_disc": (10, 60),
        "SR_MM_disc": (0, 40),
        "SR_HM_disc": (0, 30),
    }

    region_name_to_anchors = {
        "SR_MM_disc": [0.0, 5.0],
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
