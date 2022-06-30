"""
time python searches/atlas_susy_2Ljets_2022/dump_fit_signal.py

"""

import os

from pyhf_stuff import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "ewk_high": (0, 10),
        "ewk_int": (0, 40),
        "ewk_llbb": (0, 10),
        "ewk_low": (0, 30),
        "ewk_offshell": (0, 30),
        "rjr_sr2l_isr": (0, 40),
        "rjr_sr2l_low": (0, 40),
        "str_src_12_31": (0, 15),
        "str_src_12_61": (0, 20),
        "str_src_31_81": (0, 25),
        "str_src_81": (0, 40),
        "str_srhigh_12_301": (0, 25),
        "str_srhigh_301": (0, 12),
        "str_srlow_101_201": (0, 25),
        "str_srlow_101_301": (0, 35),
        "str_srlow_12_81": (0, 30),
        "str_srlow_301": (0, 20),
        "str_srmed_101": (0, 40),
        "str_srmed_12_101": (0, 40),
        "str_srzhigh": (0, 15),
        "str_srzlow": (0, 40),
        "str_srzmed": (0, 30),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=200):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_signal.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
