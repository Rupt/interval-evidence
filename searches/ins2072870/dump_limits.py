"""
time python searches/ins2072870/dump_limits.py

"""

import os

from pyhf_stuff import (
    fit_cabinetry,
    fit_linspace,
    fit_mcmc_mix,
    fit_normal,
    fit_signal,
    limit,
    models,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    region_name_to_scan = {
        "ewk_high": (0, 10),
        "ewk_int": (0, 40),
        "ewk_llbb": (0, 10),
        "ewk_low": (0, 30),
        "ewk_offshell": (0, 30),
        "rjr_sr2l_isr": (0, 40),
        "rjr_sr2l_low": (0, 40),
        "str_src_12_31": (0, 15),
        "str_src_12_61": (0, 20),
        "str_src_31_81": (0, 25),
        "str_src_81": (0, 40),
        "str_srhigh_12_301": (0, 25),
        "str_srhigh_301": (0, 12),
        "str_srlow_101_201": (0, 25),
        "str_srlow_101_301": (0, 35),
        "str_srlow_12_81": (0, 30),
        "str_srlow_301": (0, 20),
        "str_srmed_101": (0, 40),
        "str_srmed_12_101": (0, 40),
        "str_srzhigh": (0, 15),
        "str_srzlow": (0, 40),
        "str_srzmed": (0, 30),
    }

    for name, (lo, hi) in region_name_to_scan.items():
        print(name)
        dump_region(name, lo, hi)


def dump_region(name, lo, hi):
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

    # normal
    fit = fit_normal.FitNormal.load(path_fit)
    dump(fit.filename, fit, models.normal)
    dump(fit.filename + "_log", fit, models.normal_log)

    # linspace
    fit = fit_linspace.FitLinspace.load(path_fit)
    dump(fit.filename, fit, models.linspace)

    # mcmc
    fit = fit_mcmc_mix.FitMcmcMix.load(path_fit)
    dump(fit.filename, fit, models.mcmc)

    # fit signal scan
    signal = fit_signal.FitSignal.load(path_fit)
    limit.dump_scan_fit_signal(signal.filename, signal, path_limit)


if __name__ == "__main__":
    main()
