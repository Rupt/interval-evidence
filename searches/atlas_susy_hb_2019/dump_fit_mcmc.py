"""
time python searches/atlas_susy_hb_2019/dump_fit_mcmc.py

"""

import os

import numpy

from discohist import fit_mcmc_mix, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SRA": (0, 40),
        "SRB": (0, 20),
        "SRC": (0, 100),
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
