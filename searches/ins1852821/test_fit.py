"""
time python searches/ins1852821/test_fit.py

"""

import os

import numpy

from pyhf_stuff import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_linspace,
    fit_mcmc_mix,
    fit_normal,
    mcmc_core,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR0breq": (0, 5),
        "SR0bvetoloose": (0, 50),
        "SR0bvetotight": (0, 25),
        "SR0ZZbvetoloose": (0, 40),
        "SR0ZZbvetotight": (0, 7),
        "SR0ZZloose": (50, 500),
        "SR0ZZtight": (5, 50),
        "SR1breq": (0, 11),
        "SR1bvetoloose": (0, 30),
        "SR1bvetotight": (0, 15),
        "SR2breq": (0, 20),
        "SR2bvetoloose": (0, 100),
        "SR2bvetotight": (0, 20),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=25):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_cabinetry.fit(region_1).dump(dir_fit)
    fit_cabinetry_post.fit(region_1).dump(dir_fit)
    fit_normal.fit(region_1).dump(dir_fit)

    fit_linspace.fit(region_1, lo, hi, nbins + 1).dump(dir_fit)

    mala = fit_mcmc_mix.fit(
        region_1,
        nbins,
        (lo, hi),
        seed=0,
        nsamples=20_000,
        nrepeats=16,
        nprocesses=8,
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
