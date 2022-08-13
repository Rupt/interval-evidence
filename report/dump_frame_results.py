"""Dump a frame of results extracted from the many small files in searches/.

Usage:

python report/dump_frame_results.py

"""
import functools
import json
import os
from types import SimpleNamespace

import frame
import numpy

from discohisto import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_normal,
    limit,
    region,
    stats,
)

SEARCHES_PATH = "searches/"
RESULTS_PATH = "report/results.csv"


def main():
    frame_ = load_frame()
    frame.dump(frame_, RESULTS_PATH)


def load_frame():
    searches = load_searches()

    # frame entries
    search_ = []
    region_ = []
    reported_n = []
    reported_bkg = []
    reported_bkg_hi = []
    reported_bkg_lo = []
    reported_s95obs = []
    reported_s95exp = []
    reported_s95exp_hi = []
    reported_s95exp_lo = []

    region_n = []

    # fit results
    fit_cabinetry_bkg = []
    fit_cabinetry_err = []
    fit_cabinetry_post_bkg = []
    fit_cabinetry_post_err = []

    # limits observed
    limit_cabinetry_logl = []
    limit_cabinetry_2obs = []
    limit_cabinetry_3obs = []

    limit_cabinetry_post_2obs = []
    limit_cabinetry_post_3obs = []

    limit_normal_logl = []
    limit_normal_2obs = []
    limit_normal_3obs = []

    limit_normal_log_logl = []
    limit_normal_log_2obs = []
    limit_normal_log_3obs = []

    limit_linspace_logl = []
    limit_linspace_2obs = []
    limit_linspace_3obs = []

    limit_delta_logl = []
    limit_delta_2obs = []
    limit_delta_3obs = []

    limit_mcmc_logl = []
    limit_mcmc_2obs = []
    limit_mcmc_3obs = []

    # limits expected and corresponding data
    limit_cabinetry_3exp = []
    limit_cabinetry_3exp_hi = []
    limit_cabinetry_3exp_lo = []
    limit_cabinetry_nexp = []
    limit_cabinetry_nexp_hi = []
    limit_cabinetry_nexp_lo = []

    limit_normal_3exp = []
    limit_normal_3exp_hi = []
    limit_normal_3exp_lo = []
    limit_normal_nexp = []
    limit_normal_nexp_hi = []
    limit_normal_nexp_lo = []

    limit_normal_log_3exp = []
    limit_normal_log_3exp_hi = []
    limit_normal_log_3exp_lo = []
    limit_normal_log_nexp = []
    limit_normal_log_nexp_hi = []
    limit_normal_log_nexp_lo = []

    limit_linspace_3exp = []
    limit_linspace_3exp_hi = []
    limit_linspace_3exp_lo = []
    limit_linspace_nexp = []
    limit_linspace_nexp_hi = []
    limit_linspace_nexp_lo = []

    limit_delta_3exp = []
    limit_delta_3exp_hi = []
    limit_delta_3exp_lo = []
    limit_delta_nexp = []
    limit_delta_nexp_hi = []
    limit_delta_nexp_lo = []

    limit_mcmc_3exp = []
    limit_mcmc_3exp_hi = []
    limit_mcmc_3exp_lo = []
    limit_mcmc_nexp = []
    limit_mcmc_nexp_hi = []
    limit_mcmc_nexp_lo = []

    for search in searches:
        reported = load_reported(search)
        for region_name in reported:
            search_.append(search)
            region_.append(region_name)

            # reported
            reported_reg = reported[region_name]

            n_observed = reported_reg["n"]
            reported_n.append(n_observed)
            reported_bkg.append(reported_reg["bkg"])
            reported_bkg_hi.append(reported_reg["bkg_hi"])
            reported_bkg_lo.append(reported_reg["bkg_lo"])
            reported_s95obs.append(reported_reg["s95obs"])
            reported_s95exp.append(reported_reg["s95exp"])
            reported_s95exp_hi.append(reported_reg["s95exp_hi"])
            reported_s95exp_lo.append(reported_reg["s95exp_lo"])

            # region
            region_dir = os.path.join(SEARCHES_PATH, search, region_name)
            region_i = region.Region.load(region_dir)
            region_n.append(_get_n_region(region_i))

            # standard fits
            fit_dir = os.path.join(region_dir, "fit")

            fit = fit_cabinetry.FitCabinetry.load(fit_dir)
            fit_cabinetry_bkg.append(fit.yield_pre)
            fit_cabinetry_err.append(fit.error_pre)

            fit = fit_cabinetry_post.FitCabinetryPost.load(fit_dir)
            fit_cabinetry_post_bkg.append(fit.yield_post)
            fit_cabinetry_post_err.append(fit.error_post)

            fit = fit_normal.FitNormal.load(fit_dir)
            mu_delta = fit.yield_linear

            # limits
            limit_dir = os.path.join(fit_dir, "limit")
            load_limit = functools.partial(_load_limit, limit_dir)

            # limits observed
            lim = load_limit("_cabinetry_observed")
            limit_cabinetry_logl.append(_limit_logl(lim))
            limit_cabinetry_2obs.append(last2(lim))
            limit_cabinetry_3obs.append(last3(lim))

            lim = load_limit("_cabinetry_post_observed")
            limit_cabinetry_post_2obs.append(last2(lim))
            limit_cabinetry_post_3obs.append(last3(lim))

            lim = load_limit("_normal_observed")
            limit_normal_logl.append(_limit_logl(lim))
            limit_normal_2obs.append(last2(lim))
            limit_normal_3obs.append(last3(lim))

            lim = load_limit("_normal_log_observed")
            limit_normal_log_logl.append(_limit_logl(lim))
            limit_normal_log_2obs.append(last2(lim))
            limit_normal_log_3obs.append(last3(lim))

            lim = load_limit("_linspace_observed")
            limit_linspace_logl.append(_limit_logl(lim))
            limit_linspace_2obs.append(last2(lim))
            limit_linspace_3obs.append(last3(lim))

            lim = limit.LimitScanDelta.load(limit_dir, suffix="_observed")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_delta_logl.append(
                stats.poisson_log_minus_max(n_observed, mu_delta)
            )
            limit_delta_2obs.append(last2(lim))
            limit_delta_3obs.append(last3(lim))

            lim = _load_mcmc_limits(limit_dir, suffix="observed")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_mcmc_logl.append(_limit_logl(lim))
            limit_mcmc_2obs.append(last2(lim))
            limit_mcmc_3obs.append(last3(lim))

            # limits expected
            lim = load_limit("_cabinetry_central")
            limit_cabinetry_3exp.append(last3(lim))
            limit_cabinetry_nexp.append(lim.ndata)
            lim = load_limit("_cabinetry_up")
            limit_cabinetry_3exp_hi.append(last3(lim))
            limit_cabinetry_nexp_hi.append(lim.ndata)
            lim = load_limit("_cabinetry_down")
            limit_cabinetry_3exp_lo.append(last3(lim))
            limit_cabinetry_nexp_lo.append(lim.ndata)

            lim = load_limit("_normal_central")
            limit_normal_3exp.append(last3(lim))
            limit_normal_nexp.append(lim.ndata)
            lim = load_limit("_normal_up")
            limit_normal_3exp_hi.append(last3(lim))
            limit_normal_nexp_hi.append(lim.ndata)
            lim = load_limit("_normal_down")
            limit_normal_3exp_lo.append(last3(lim))
            limit_normal_nexp_lo.append(lim.ndata)

            lim = load_limit("_normal_log_central")
            limit_normal_log_3exp.append(last3(lim))
            limit_normal_log_nexp.append(lim.ndata)
            lim = load_limit("_normal_log_up")
            limit_normal_log_3exp_hi.append(last3(lim))
            limit_normal_log_nexp_hi.append(lim.ndata)
            lim = load_limit("_normal_log_down")
            limit_normal_log_3exp_lo.append(last3(lim))
            limit_normal_log_nexp_lo.append(lim.ndata)

            lim = load_limit("_linspace_central")
            limit_linspace_3exp.append(last3(lim))
            limit_linspace_nexp.append(lim.ndata)
            lim = load_limit("_linspace_up")
            limit_linspace_3exp_hi.append(last3(lim))
            limit_linspace_nexp_hi.append(lim.ndata)
            lim = load_limit("_linspace_down")
            limit_linspace_3exp_lo.append(last3(lim))
            limit_linspace_nexp_lo.append(lim.ndata)

            lim = limit.LimitScanDelta.load(limit_dir, suffix="_central")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_delta_3exp.append(last3(lim))
            limit_delta_nexp.append(lim.ndata)
            lim = limit.LimitScanDelta.load(limit_dir, suffix="_up")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_delta_3exp_hi.append(last3(lim))
            limit_delta_nexp_hi.append(lim.ndata)
            lim = limit.LimitScanDelta.load(limit_dir, suffix="_down")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_delta_3exp_lo.append(last3(lim))
            limit_delta_nexp_lo.append(lim.ndata)

            lim = _load_mcmc_limits(limit_dir, suffix="central")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_mcmc_3exp.append(last3(lim))
            limit_mcmc_nexp.append(lim.ndata)
            lim = _load_mcmc_limits(limit_dir, suffix="up")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_mcmc_3exp_hi.append(last3(lim))
            limit_mcmc_nexp_hi.append(lim.ndata)
            lim = _load_mcmc_limits(limit_dir, suffix="down")
            assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
            limit_mcmc_3exp_lo.append(last3(lim))
            limit_mcmc_nexp_lo.append(lim.ndata)

    out = dict(
        # labels
        search_=search_,
        region_=region_,
        # reported
        reported_n=reported_n,
        reported_bkg=reported_bkg,
        reported_bkg_hi=reported_bkg_hi,
        reported_bkg_lo=reported_bkg_lo,
        reported_s95obs=reported_s95obs,
        reported_s95exp=reported_s95exp,
        reported_s95exp_hi=reported_s95exp_hi,
        reported_s95exp_lo=reported_s95exp_lo,
        region_n=region_n,
        # fits
        fit_cabinetry_bkg=fit_cabinetry_bkg,
        fit_cabinetry_err=fit_cabinetry_err,
        fit_cabinetry_post_bkg=fit_cabinetry_post_bkg,
        fit_cabinetry_post_err=fit_cabinetry_post_err,
        # limits
        # observed limits
        limit_cabinetry_logl=limit_cabinetry_logl,
        limit_cabinetry_2obs=limit_cabinetry_2obs,
        limit_cabinetry_3obs=limit_cabinetry_3obs,
        limit_cabinetry_post_2obs=limit_cabinetry_post_2obs,
        limit_cabinetry_post_3obs=limit_cabinetry_post_3obs,
        limit_normal_logl=limit_normal_logl,
        limit_normal_2obs=limit_normal_2obs,
        limit_normal_3obs=limit_normal_3obs,
        limit_normal_log_logl=limit_normal_log_logl,
        limit_normal_log_2obs=limit_normal_log_2obs,
        limit_normal_log_3obs=limit_normal_log_3obs,
        limit_linspace_logl=limit_linspace_logl,
        limit_linspace_2obs=limit_linspace_2obs,
        limit_linspace_3obs=limit_linspace_3obs,
        limit_delta_logl=limit_delta_logl,
        limit_delta_2obs=limit_delta_2obs,
        limit_delta_3obs=limit_delta_3obs,
        limit_mcmc_logl=limit_mcmc_logl,
        limit_mcmc_2obs=limit_mcmc_2obs,
        limit_mcmc_3obs=limit_mcmc_3obs,
        # expected limits
        limit_cabinetry_3exp=limit_cabinetry_3exp,
        limit_cabinetry_3exp_hi=limit_cabinetry_3exp_hi,
        limit_cabinetry_3exp_lo=limit_cabinetry_3exp_lo,
        limit_cabinetry_nexp=limit_cabinetry_nexp,
        limit_cabinetry_nexp_hi=limit_cabinetry_nexp_hi,
        limit_cabinetry_nexp_lo=limit_cabinetry_nexp_lo,
        limit_normal_3exp=limit_normal_3exp,
        limit_normal_3exp_hi=limit_normal_3exp_hi,
        limit_normal_3exp_lo=limit_normal_3exp_lo,
        limit_normal_nexp=limit_normal_nexp,
        limit_normal_nexp_hi=limit_normal_nexp_hi,
        limit_normal_nexp_lo=limit_normal_nexp_lo,
        limit_normal_log_3exp=limit_normal_log_3exp,
        limit_normal_log_3exp_hi=limit_normal_log_3exp_hi,
        limit_normal_log_3exp_lo=limit_normal_log_3exp_lo,
        limit_normal_log_nexp=limit_normal_log_nexp,
        limit_normal_log_nexp_hi=limit_normal_log_nexp_hi,
        limit_normal_log_nexp_lo=limit_normal_log_nexp_lo,
        limit_linspace_3exp=limit_linspace_3exp,
        limit_linspace_3exp_hi=limit_linspace_3exp_hi,
        limit_linspace_3exp_lo=limit_linspace_3exp_lo,
        limit_linspace_nexp=limit_linspace_nexp,
        limit_linspace_nexp_hi=limit_linspace_nexp_hi,
        limit_linspace_nexp_lo=limit_linspace_nexp_lo,
        limit_delta_3exp=limit_delta_3exp,
        limit_delta_3exp_hi=limit_delta_3exp_hi,
        limit_delta_3exp_lo=limit_delta_3exp_lo,
        limit_delta_nexp=limit_delta_nexp,
        limit_delta_nexp_hi=limit_delta_nexp_hi,
        limit_delta_nexp_lo=limit_delta_nexp_lo,
        limit_mcmc_3exp=limit_mcmc_3exp,
        limit_mcmc_3exp_hi=limit_mcmc_3exp_hi,
        limit_mcmc_3exp_lo=limit_mcmc_3exp_lo,
        limit_mcmc_nexp=limit_mcmc_nexp,
        limit_mcmc_nexp_hi=limit_mcmc_nexp_hi,
        limit_mcmc_nexp_lo=limit_mcmc_nexp_lo,
    )

    return SimpleNamespace(
        **{key: numpy.array(value) for key, value in out.items()}
    )


# utilities


@functools.cache
def load_searches():
    searches = []
    for item in os.scandir(SEARCHES_PATH):
        if not item.is_dir():
            continue
        searches.append(item.name)

    return sorted(searches)


def load_reported(search):
    path = os.path.join(SEARCHES_PATH, search, "reported.json")
    with open(path) as file_:
        reported = json.load(file_)
    return reported


def _get_n_region(reg):
    sr_name = reg.signal_region_name

    for obs in reg.workspace["observations"]:
        if obs["name"] == sr_name:
            return obs["data"][0]

    raise ValueError(sr_name)


def _load_mcmc_limits(path, *, suffix):
    # different mcmc strategies have different filenames
    # we currently use mix and tfp ham
    mcmc_types = ["mix", "tfp_ham"]
    lim = None
    for mcmc_type in mcmc_types:
        suffix_i = "_mcmc_%s_%s" % (mcmc_type, suffix)
        try:
            lim = limit.LimitScan.load(path, suffix=suffix_i)
        except FileNotFoundError:
            ...
    assert lim is not None
    return lim


def _load_limit(limit_dir, suffix):
    lim = limit.LimitScan.load(limit_dir, suffix=suffix)
    assert lim.levels[6:8] == [-2, -3], lim.levels[6:8]
    return lim


def _limit_logl(lim):
    return numpy.log(numpy.mean(lim.integral_zero))


def last2(limit):
    return limit.points[6][-1]


def last3(limit):
    return limit.points[7][-1]


if __name__ == "__main__":
    main()
