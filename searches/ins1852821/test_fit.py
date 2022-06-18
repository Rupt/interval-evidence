"""
time python searches/ins1852821/test_fit.py

"""

import os

from pyhf_stuff import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_interval,
    fit_linspace,
    fit_mcmc_mala,
    fit_normal,
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

    dir_fit = os.path.join(dir_region, "fit")

    fit_cabinetry.fit(region_1).dump(dir_fit)
    fit_cabinetry_post.fit(region_1).dump(dir_fit)
    fit_normal.fit(region_1).dump(dir_fit)

    interval = fit_interval.fit(region_1)
    interval.dump(dir_fit)
    # default index 3 -> 4 sigma
    lo, hi = interval.intervals[3]
    lo = max(lo, 0.0)

    linspace = fit_linspace.fit(region_1, lo, hi, nbins + 1)
    linspace.dump(dir_fit)

    mala = fit_mcmc_mala.fit(
        region_1,
        nbins,
        (lo, hi),
        seed=0,
        nsamples=100_000,
        nrepeats=16,
        nprocesses=8,
    )
    mala.dump(dir_fit)

    neff = mcmc_core.n_by_fit(mala).sum()
    print(mala.nsamples, neff, neff / mala.nsamples)


if __name__ == "__main__":
    main()
