"""
time python searches/ins2072870/dump_fit_mcmc.py

"""

import os

import numpy

from pyhf_stuff import fit_mcmc_mix, mcmc_core, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "ewk_high": (2, 8),
        "ewk_int": (20, 70),
        "ewk_llbb": (0, 3),
        "ewk_low": (0, 100),
        "ewk_offshell": (0, 40),
        "rjr_sr2l_isr": (0, 100),
        "rjr_sr2l_low": (0, 100),
        "str_src_12_31": (0, 20),
        "str_src_12_61": (0, 40),
        "str_src_31_81": (0, 40),
        "str_src_81": (0, 100),
        "str_srhigh_12_301": (0, 60),
        "str_srhigh_301": (0, 20),
        "str_srlow_101_201": (0, 80),
        "str_srlow_101_301": (0, 80),
        "str_srlow_12_81": (0, 35),
        "str_srlow_301": (0, 20),
        "str_srmed_101": (0, 100),
        "str_srmed_12_101": (20, 90),
        "str_srzhigh": (0, 20),
        "str_srzlow": (0, 60),
        "str_srzmed": (0, 40),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi, nbins=50):
    dir_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    # we explore the rjr models very poorly
    if name.startswith("rjr"):
        nsamples = 10 * 100_000
        step_size = 0.1
        prob_eye = 0.01
    else:
        nsamples = 100_000
        step_size = 0.5
        prob_eye = 0.1

    mix = fit_mcmc_mix.fit(
        region_1,
        nbins,
        (lo, hi),
        seed=0,
        nsamples=nsamples,
        nrepeats=100,
        step_size=step_size,
        prob_eye=prob_eye,
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
