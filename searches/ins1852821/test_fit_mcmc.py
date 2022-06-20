"""
time python searches/ins1852821/test_fit_mcmc.py

"""

import os

import numpy

from pyhf_stuff import fit_mcmc_mix, mcmc_core, region

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
        "SR2breq": (0, 10),
        "SR2bvetoloose": (0, 50),
        "SR2bvetotight": (0, 10),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    mala = fit_mcmc_mix.fit(
        region_1,
        nbins,
        (lo, hi),
        seed=0,
        nsamples=100_000,
        nrepeats=100,
        nprocesses=10,
    )
    mala.dump(dir_fit)

    neff = mcmc_core.n_by_fit(mala).sum()
    nrepeats = mala.nrepeats
    nsamples = mala.nsamples
    total = numpy.sum(mala.yields)
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
