"""
time python searches/atlas_susy_jets_2021/dump_fit_signal.py

"""

import os

from discohist import fit_signal, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "BDT-GGd1": (0, 40),
        "BDT-GGd2": (0, 80),
        "BDT-GGd3": (0, 100),
        "BDT-GGd4": (0, 100),
        "BDT-GGo1": (0, 20),
        "BDT-GGo2": (0, 40),
        "BDT-GGo3": (0, 50),
        "BDT-GGo4": (0, 60),
        "SR2j-1600": (0, 400),
        "SR2j-2200": (0, 200),
        "SR2j-2800": (0, 40),
        "SR4j-1000": (0, 150),
        "SR4j-2200": (0, 40),
        "SR4j-3400": (0, 15),
        "SR5j-1600": (0, 100),
        "SR6j-1000": (0, 30),
        "SR6j-2200": (0, 15),
        "SR6j-3400": (0, 10),
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
