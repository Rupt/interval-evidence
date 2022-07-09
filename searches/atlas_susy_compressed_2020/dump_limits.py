"""
time python searches/atlas_susy_compressed_2020/dump_limits.py

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
        "SR_E_mll_1": (0, 15),
        "SR_E_mll_2": (0, 80),
        "SR_E_mll_3": (0, 100),
        "SR_E_mll_5": (0, 150),
        "SR_E_mll_10": (0, 200),
        "SR_E_mll_20": (0, 300),
        "SR_E_mll_30": (0, 300),
        "SR_E_mll_40": (0, 300),
        "SR_E_mll_60": (0, 300),
        "SR_S_100p5": (0, 60),
        "SR_S_101": (0, 110),
        "SR_S_102": (0, 100),
        "SR_S_105": (0, 150),
        "SR_S_110": (0, 150),
        "SR_S_120": (0, 200),
        "SR_S_130": (0, 200),
        "SR_S_140": (0, 250),
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
    if name.startswith("SR_E"):
        fit = fit_mcmc_mix.FitMcmcMix.load(path_fit)
    else:
        fit = fit_mcmc_tfp_ham.FitMcmcTfpHam.load(path_fit)
    dump(fit.filename, fit, models.mcmc)

    # fit signal scan
    signal = fit_signal.FitSignal.load(path_fit)
    limit.dump_scan_fit_signal(
        signal.filename, signal, path_limit, print_=True
    )


if __name__ == "__main__":
    main()
