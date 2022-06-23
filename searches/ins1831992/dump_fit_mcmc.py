"""
time python searches/ins1831992/dump_fit_mcmc.py

"""

import os

import numpy

from pyhf_stuff import fit_mcmc_mix, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        # SRFR
        "SRFR_90_110_all": (0, 5),
        "SRFR_110_130_all": (0, 15),
        "SRFR_150_170_all": (0, 15),
        "SRFR_170_190_all": (0, 10),
        "SRFR_190_210_all": (0, 10),
        "SRFR_210_230_all": (0, 50),
        "SRFR_230_250_all": (0, 5),
        "SRFR_250_270_all": (0, 5),
        "SRFR_270_300_all": (0, 8),
        "SRFR_300_330_all": (0, 3),
        "SRFR_330_360_all": (0, 4),
        "SRFR_360_400_all": (0, 8),
        "SRFR_400_440_all": (0, 2),
        "SRFR_440_580_all": (0, 8),
        "SRFR_580_inf_all": (0, 10),
        # SR4L
        "SR4l_90_110_all": (0, 13),
        "SR4l_110_130_all": (10, 25),
        "SR4l_150_170_all": (5, 15),
        "SR4l_170_190_all": (0, 12),
        "SR4l_190_210_all": (0, 12),
        "SR4l_210_230_all": (0, 6),
        "SR4l_230_250_all": (0, 6),
        "SR4l_250_270_all": (0, 8),
        "SR4l_270_300_all": (0, 4),
        "SR4l_300_330_all": (0, 3),
        "SR4l_330_360_all": (0, 8),
        "SR4l_360_400_all": (0, 8),
        "SR4l_400_440_all": (0, 8),
        "SR4l_440_580_all": (0, 8),
        "SR4l_580_inf_all": (0, 10),
        # SR3L
        "SR3l_90_110_all": (0, 3),
        "SR3l_110_130_all": (0, 6),
        "SR3l_150_170_all": (0, 10),
        "SR3l_170_190_all": (0, 7),
        "SR3l_190_210_all": (0, 7),
        "SR3l_210_230_all": (0, 10),
        "SR3l_230_250_all": (0, 10),
        "SR3l_250_270_all": (0, 5),
        "SR3l_270_300_all": (0, 7),
        "SR3l_300_330_all": (0, 7),
        "SR3l_330_360_all": (0, 8),
        "SR3l_360_400_all": (0, 8),
        "SR3l_400_440_all": (0, 4),
        "SR3l_440_580_all": (0, 8),
        "SR3l_580_inf_all": (0, 8),
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
        nprocesses=10,
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
