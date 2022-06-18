"""
time python searches/ins1852821/test_fit.py

"""

import os

from pyhf_stuff import (
    fit_cabinetry,
    fit_cabinetry_post,
    fit_interval,
    fit_linspace,
    fit_mcmc_mala,
    fit_mcmc_tfp_ham,
    fit_mcmc_tfp_mala,
    fit_mcmc_tfp_nuts,
    fit_normal,
    mcmc_core,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    dir_region = os.path.join(BASEPATH, "SR0bvetotight")
    region_1 = region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    fit_cabinetry.fit(region_1).dump(dir_fit)
    fit_cabinetry_post.fit(region_1).dump(dir_fit)
    fit_normal.fit(region_1).dump(dir_fit)

    return

    fit_int = fit_interval.fit(region_1)
    fit_interval.dump(fit_int, dir_fit)
    print(fit_int)
    assert fit_int == fit_interval.load(dir_fit)

    fit_lin = fit_linspace.fit(region_1, 0, 25, 26)
    fit_linspace.dump(fit_lin, dir_fit)
    print(fit_lin)
    assert fit_lin == fit_linspace.load(dir_fit)

    # mcmc
    fit_mal = fit_mcmc_tfp_mala.fit(
        region_1,
        25,
        (0.0, 25.0),
        seed=0,
        nsamples=10_000,
        nrepeats=8,
    )
    fit_mcmc_tfp_mala.dump(fit_mal, dir_fit)
    print(fit_mal)
    assert fit_mal == fit_mcmc_tfp_mala.load(dir_fit)
    fit_mal = fit_mcmc_tfp_mala.load(dir_fit)
    neff = mcmc_core.n_by_fit(fit_mal).sum()
    print(fit_mal.nsamples, neff, neff / fit_mal.nsamples)

    fit_ham = fit_mcmc_tfp_ham.fit(
        region_1,
        25,
        (0.0, 25.0),
        seed=0,
        nsamples=10_000,
        nrepeats=8,
    )
    fit_mcmc_tfp_ham.dump(fit_ham, dir_fit)
    print(fit_ham)
    assert fit_ham == fit_mcmc_tfp_ham.load(dir_fit)
    neff = mcmc_core.n_by_fit(fit_ham).sum()
    print(fit_ham.nsamples, neff, neff / fit_ham.nsamples)

    fit_nuts = fit_mcmc_tfp_nuts.fit(
        region_1,
        25,
        (0.0, 25.0),
        seed=0,
        nsamples=10_000,
        nrepeats=8,
    )
    fit_mcmc_tfp_nuts.dump(fit_nuts, dir_fit)
    print(fit_nuts)
    assert fit_nuts == fit_mcmc_tfp_nuts.load(dir_fit)
    neff = mcmc_core.n_by_fit(fit_nuts).sum()
    print(fit_ham.nsamples, neff, neff / fit_ham.nsamples)

    fit_mal2 = fit_mcmc_mala.fit(
        region_1,
        10,
        (0.0, 25.0),
        seed=1,
        nsamples=10_000,
        nrepeats=8,
        nprocesses=8,
    )
    fit_mcmc_mala.dump(fit_mal2, dir_fit)
    print(fit_mal2)
    assert fit_mal2 == fit_mcmc_mala.load(dir_fit)
    fit_mal = fit_mcmc_mala.load(dir_fit)
    neff = mcmc_core.n_by_fit(fit_mal2).sum()
    print(fit_mal2.nsamples, neff, neff / fit_mal2.nsamples)


if __name__ == "__main__":
    main()
