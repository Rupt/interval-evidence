"""
time python searches/atlas_susy_2Ljets_2022/dump_limits.py

"""

import os

from discohist import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_linspace,
    fit_mcmc_mix,
    fit_mcmc_tfp_ham,
    fit_normal,
    fit_signal,
    limit,
    models,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "ewk_high": (0, 20),
        "ewk_int": (0, 40),
        "ewk_llbb": (0, 10),
        "ewk_low": (0, 80),
        "ewk_offshell": (0, 30),
        "rjr_sr2l_isr": (0, 60),
        "rjr_sr2l_low": (0, 60),
        "str_src_12_31": (0, 25),
        "str_src_12_61": (0, 25),
        "str_src_31_81": (0, 40),
        "str_src_81": (0, 80),
        "str_srhigh_12_301": (0, 40),
        "str_srhigh_301": (0, 20),
        "str_srlow_101_201": (0, 50),
        "str_srlow_101_301": (0, 80),
        "str_srlow_12_81": (0, 30),
        "str_srlow_301": (0, 30),
        "str_srmed_101": (0, 50),
        "str_srmed_12_101": (0, 50),
        "str_srzhigh": (0, 20),
        "str_srzlow": (0, 40),
        "str_srzmed": (0, 30),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_limits(name, lo, hi)


def dump_limits(name, lo, hi):
    path_region = os.path.join(BASEPATH, name)
    region_1 = region.Region.load(path_region)

    path_fit = os.path.join(path_region, "fit")
    path_limit = os.path.join(path_fit, "limit")

    def dump(label, fit, model_fn):
        return limit.dump_scans(
            label,
            fit,
            model_fn,
            path_limit,
            region_1.ndata,
            lo,
            hi,
            print_=True,
        )

    # cabinetry
    fit = fit_cabinetry.FitCabinetry.load(path_fit)
    dump(fit.filename, fit, models.cabinetry)

    fit = fit_cabinetry_post.FitCabinetryPost.load(path_fit)
    dump(fit.filename, fit, models.cabinetry_post)

    # normal
    fit = fit_normal.FitNormal.load(path_fit)
    dump(fit.filename, fit, models.normal)
    dump(fit.filename + "_log", fit, models.normal_log)

    # best fit only, no uncertainty
    limit.dump_scan_delta(
        "delta",
        fit.yield_linear,
        path_limit,
        region_1.ndata,
        lo,
        hi,
        print_=True,
    )

    # linspace
    fit = fit_linspace.FitLinspace.load(path_fit)
    dump(fit.filename, fit, models.linspace)

    # mcmc
    if name.startswith("rjr"):
        fit = fit_mcmc_tfp_ham.FitMcmcTfpHam.load(path_fit)
        dump(fit.filename, fit, models.mcmc)
    else:
        fit = fit_mcmc_mix.FitMcmcMix.load(path_fit)
        dump(fit.filename, fit, models.mcmc)

    # fit signal scan
    signal = fit_signal.FitSignal.load(path_fit)
    limit.dump_scan_fit_signal(
        signal.filename, signal, path_limit, print_=True
    )


if __name__ == "__main__":
    main()
