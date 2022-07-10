"""
time python searches/atlas_susy_compressed_2020/dump_fit_mcmc.py

"""

import os

import numpy

from discohisto import fit_mcmc_mix, fit_mcmc_tfp_ham, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "SR_E_mll_1": (0, 6),
        "SR_E_mll_2": (10, 100),
        "SR_E_mll_3": (20, 150),
        "SR_E_mll_5": (80, 220),
        "SR_E_mll_10": (120, 300),
        "SR_E_mll_20": (200, 450),
        "SR_E_mll_30": (250, 550),
        "SR_E_mll_40": (300, 600),
        "SR_E_mll_60": (350, 700),
        "SR_S_100p5": (10, 120),
        "SR_S_101": (25, 200),
        "SR_S_102": (45, 250),
        "SR_S_105": (100, 300),
        "SR_S_110": (150, 400),
        "SR_S_120": (200, 460),
        "SR_S_130": (250, 600),
        "SR_S_140": (300, 650),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_mcmc(name, lo, hi)


def dump_mcmc(name, lo, hi, nbins=200):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    if name == "SR_E_mll_40":
        # this one got 0.00 efficiency otherwise
        result = fit_mcmc_mix.fit(
            region_1,
            nbins,
            (lo, hi),
            seed=0,
            nsamples=100_000,
            nrepeats=100,
            step_size=0.2,
        )
    elif name.startswith("SR_E"):
        result = fit_mcmc_mix.fit(
            region_1,
            nbins,
            (lo, hi),
            seed=0,
            nsamples=100_000,
            nrepeats=100,
        )
    else:
        result = fit_mcmc_tfp_ham.fit(
            region_1,
            nbins,
            (lo, hi),
            seed=0,
            nsamples=100_000,
            nrepeats=100,
            step_size=0.05,
            num_leapfrog_steps=10,
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
