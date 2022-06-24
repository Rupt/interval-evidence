"""
time python searches/ins1827025/dump_fit_mcmc.py

"""

import os

import numpy

from pyhf_stuff import fit_mcmc_mix, mcmc_core, region

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
        "SR6j-2200": (0, 30),
        "SR6j-3400": (0, 10),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    mix = fit_mcmc_mix.fit(
        region_1,
        nbins,
        (lo, hi),
        seed=0,
        nsamples=100_000,
        nrepeats=100,
    )
    mix.dump(dir_fit)

    neff = mcmc_core.n_by_fit(mix).sum()
    nrepeats = mix.nrepeats
    nsamples = mix.nsamples
    total = numpy.sum(mix.yields)
    print(
        "acceptance: %.2f (%d / %d)"
        % (total / (nrepeats * nsamples), total, nrepeats * nsamples)
    )
    print(
        "efficiency: %.2f (%.1f / %.1f)"
        % (nrepeats * neff / total, neff, total / nrepeats)
    )


if __name__ == "__main__":
    main()
