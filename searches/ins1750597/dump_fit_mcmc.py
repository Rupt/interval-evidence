"""
time python searches/ins1750597/dump_fit_mcmc.py

"""

import os

import numpy

from pyhf_stuff import fit_mcmc_mix, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        # DF 0J
        "SR_DF_0J_100_inf": (50, 250),
        "SR_DF_0J_160_inf": (10, 70),
        "SR_DF_0J_100_120": (20, 140),
        "SR_DF_0J_120_160": (20, 100),
        # DF 1J
        "SR_DF_1J_100_inf": (40, 140),
        "SR_DF_1J_160_inf": (10, 50),
        "SR_DF_1J_100_120": (10, 90),
        "SR_DF_1J_120_160": (10, 45),
        # SF 0J
        "SR_SF_0J_100_inf": (80, 260),
        "SR_SF_0J_160_inf": (40, 110),
        "SR_SF_0J_100_120": (15, 120),
        "SR_SF_0J_120_160": (30, 100),
        # SF 1J
        "SR_SF_1J_100_inf": (60, 240),
        "SR_SF_1J_160_inf": (30, 110),
        "SR_SF_1J_100_120": (0, 120),
        "SR_SF_1J_120_160": (25, 65),
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
