"""
time python searches/atlas_susy_4L_2021/dump_limits.py

"""

import os

from discohisto import (
    fit_cabinetry,
    fit_cabinetry_post,
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
        "SR0breq": (0, 20),
        "SR0bvetoloose": (0, 40),
        "SR0bvetotight": (0, 30),
        "SR0ZZbvetoloose": (0, 30),
        "SR0ZZbvetotight": (0, 30),
        "SR0ZZloose": (0, 250),
        "SR0ZZtight": (0, 40),
        "SR1breq": (0, 20),
        "SR1bvetoloose": (0, 30),
        "SR1bvetotight": (0, 20),
        "SR2breq": (0, 20),
        "SR2bvetoloose": (0, 35),
        "SR2bvetotight": (0, 20),
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
    fit = fit_mcmc_mix.FitMcmcMix.load(path_fit)
    dump(fit.filename, fit, models.mcmc)

    # fit signal scan
    signal = fit_signal.FitSignal.load(path_fit)
    limit.dump_scan_fit_signal(signal, path_limit, print_=True)


if __name__ == "__main__":
    main()
