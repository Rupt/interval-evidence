"""
time python searches/atlas_susy_4L_2021/dump_fit_mcmc.py

"""

import os

import numpy

from discohisto import fit_mcmc_mix, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR0breq": (0, 5),
        "SR0bvetoloose": (5, 40),
        "SR0bvetotight": (0, 35),
        "SR0ZZbvetoloose": (0, 30),
        "SR0ZZbvetotight": (0, 5),
        "SR0ZZloose": (50, 550),
        "SR0ZZtight": (5, 45),
        "SR1breq": (0, 8),
        "SR1bvetoloose": (0, 20),
        "SR1bvetotight": (0, 8),
        "SR2breq": (0, 4),
        "SR2bvetoloose": (0, 50),
        "SR2bvetotight": (0, 4),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_mcmc(name, lo, hi)


def dump_mcmc(name, lo, hi, nbins=200):
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
