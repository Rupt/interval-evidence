"""
time python searches/atlas_susy_2hadtau_2020/dump_fit_mcmc.py

"""

import os

import numpy

from discohist import fit_mcmc_tfp_ham, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SRlowMass": (0, 50),
        "SRhighMass": (0, 100),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=200):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    # these get particularly poor efficiency with mix
    result = fit_mcmc_tfp_ham.fit(
        region_1,
        nbins,
        (lo, hi),
        seed=0,
        nsamples=100_000,
        nrepeats=100,
        step_size=0.1,
        num_leapfrog_steps=5,
    )
    result.dump(dir_fit)

    neff = mcmc_core.n_by_fit(result).sum()
    nrepeats = result.nrepeats
    nsamples = result.nsamples
    total = numpy.sum(result.yields)
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
