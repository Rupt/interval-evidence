"""
time python searches/atlas_susy_1Ljets_2021/dump_fit_signal.py

"""

import os

from discohist import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR2JBVEM_meffInc30_gluino": (0, 35),
        "SR2JBVEM_meffInc30_squark": (0, 80),
        "SR4JhighxBVEM_meffInc30": (0, 40),
        "SR4JlowxBVEM_meffInc30": (0, 40),
        "SR6JBVEM_meffInc30_gluino": (0, 20),
        "SR6JBVEM_meffInc30_squark": (0, 20),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_signal(name, lo, hi)


def dump_signal(name, lo, hi, nbins=200):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_signal.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)


if __name__ == "__main__":
    main()
