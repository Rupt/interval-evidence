"""
time python searches/atlas_susy_3L_2021/dump_fit_mcmc.py

"""

import os

import numpy

from discohist import fit_mcmc_mix, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "incSR_WZ_1": (10, 70),
        "incSR_WZ_2": (0, 5),
        "incSR_WZ_5": (0, 15),
        "incSR_offWZ_highEt_nj_a": (0, 18),
        "incSR_offWZ_highEt_nj_b": (0, 8),
        "incSR_offWZ_highEt_nj_c1": (0, 25),
        "incSR_offWZ_highEt_nj_c2": (0, 10),
        "incSR_offWZ_lowEt_b": (20, 60),
        "incSR_offWZ_highEt_b": (0, 30),
        "incSR_offWZ_lowEt_c": (50, 125),
        "incSR_offWZ_highEt_c": (0, 20),
        "incSR_offWZ_d": (125, 250),
        "incSR_offWZ_e1": (200, 400),
        "incSR_offWZ_e2": (200, 350),
        "incSR_offWZ_f1": (350, 600),
        "incSR_offWZ_f2": (200, 350),
        "incSR_offWZ_g1": (450, 750),
        "incSR_offWZ_g2": (300, 500),
        "incSR_offWZ_g3": (200, 400),
        "incSR_offWZ_g4": (80, 200),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    result = fit_mcmc_mix.fit(
        region_1,
        nbins,
        (lo, hi),
        seed=0,
        nsamples=100_000,
        nrepeats=100,
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
