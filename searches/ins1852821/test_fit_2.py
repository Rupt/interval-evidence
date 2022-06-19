"""
time python searches/ins1852821/test_fit_2.py

"""

import os

from pyhf_stuff import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_interval,
    fit_linspace,
    fit_mcmc_mix,
    fit_normal,
    fit_signal,
    mcmc_core,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    region_names = [
        "SR0breq",
        "SR0bvetoloose",
        "SR0bvetotight",
        # "SR0ZZbvetoloose", # Failing to find intervals
        # "SR0ZZbvetotight", # Failing to find intervals
        # "SR0ZZloose", # Failing to find intervals
        # "SR0ZZtight", # Failing to find intervals
        "SR1breq",
        "SR1bvetoloose",
        # "SR1bvetotight", # Failing to find intervals
        "SR2breq",
        "SR2bvetoloose",
        "SR2bvetotight",
    ]

    for name in region_names:
        print(name)
        dump_region(name)


def dump_region(name):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.load(dir_region)
    nbins = 25
    lo = 0.0
    hi = 5.0

    dir_fit = os.path.join(dir_region, "fit")

    fit_signal.fit(region_1, 0, 10, 10 + 1).dump(dir_fit)



if __name__ == "__main__":
    main()
