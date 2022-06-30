"""
time python searches/atlas_susy_trilepton_2020/dump_limits.py

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
        # SRFR
        "SRFR_90_110_all": (0, 15),
        "SRFR_110_130_all": (0, 20),
        "SRFR_150_170_all": (0, 30),
        "SRFR_170_190_all": (0, 20),
        "SRFR_190_210_all": (0, 20),
        "SRFR_210_230_all": (0, 20),
        "SRFR_230_250_all": (0, 20),
        "SRFR_250_270_all": (0, 20),
        "SRFR_270_300_all": (0, 20),
        "SRFR_300_330_all": (0, 15),
        "SRFR_330_360_all": (0, 15),
        "SRFR_360_400_all": (0, 20),
        "SRFR_400_440_all": (0, 20),
        "SRFR_440_580_all": (0, 12),
        "SRFR_580_inf_all": (0, 12),
        # SR4L
        "SR4l_90_110_all": (0, 20),
        "SR4l_110_130_all": (0, 30),
        "SR4l_150_170_all": (0, 22),
        "SR4l_170_190_all": (0, 25),
        "SR4l_190_210_all": (0, 20),
        "SR4l_210_230_all": (0, 20),
        "SR4l_230_250_all": (0, 20),
        "SR4l_250_270_all": (0, 20),
        "SR4l_270_300_all": (0, 20),
        "SR4l_300_330_all": (0, 20),
        "SR4l_330_360_all": (0, 20),
        "SR4l_360_400_all": (0, 20),
        "SR4l_400_440_all": (0, 20),
        "SR4l_440_580_all": (0, 20),
        "SR4l_580_inf_all": (0, 20),
        # SR3L
        "SR3l_90_110_all": (0, 20),
        "SR3l_110_130_all": (0, 20),
        "SR3l_150_170_all": (0, 20),
        "SR3l_170_190_all": (0, 20),
        "SR3l_190_210_all": (0, 20),
        "SR3l_210_230_all": (0, 20),
        "SR3l_230_250_all": (0, 20),
        "SR3l_250_270_all": (0, 15),
        "SR3l_270_300_all": (0, 15),
        "SR3l_300_330_all": (0, 15),
        "SR3l_330_360_all": (0, 15),
        "SR3l_360_400_all": (0, 15),
        "SR3l_400_440_all": (0, 20),
        "SR3l_440_580_all": (0, 20),
        "SR3l_580_inf_all": (0, 20),
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
