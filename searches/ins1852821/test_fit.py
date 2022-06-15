"""
time python searches/ins1852821/test_fit.py

"""

import os

from pyhf_stuff import (
    fit_cabinetry,
    fit_interval,
    fit_linspace,
    fit_mcmc_ham,
    fit_mcmc_mala,
    fit_mcmc_nuts,
    fit_mymc_mala,
    fit_normal,
    mcmc,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    dir_region = os.path.join(BASEPATH, "SR0bvetotight")
    region_1 = region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

    if 0:
        fit_cab = fit_cabinetry.fit(region_1)
        fit_cabinetry.dump(fit_cab, dir_fit)
        print(fit_cab)
        assert fit_cab == fit_cabinetry.load(dir_fit)

        fit_norm = fit_normal.fit(region_1)
        fit_normal.dump(fit_norm, dir_fit)
        print(fit_norm)
        assert fit_norm == fit_normal.load(dir_fit)

        fit_int = fit_interval.fit(region_1)
        fit_interval.dump(fit_int, dir_fit)
        print(fit_int)
        assert fit_int == fit_interval.load(dir_fit)

        fit_lin = fit_linspace.fit(region_1, 0, 25, 26)
        fit_linspace.dump(fit_lin, dir_fit)
        print(fit_lin)
        assert fit_lin == fit_linspace.load(dir_fit)

    if 0:
        fit_mal = fit_mcmc_mala.fit(
            region_1,
            25,
            (0.0, 25.0),
            seed=0,
            nsamples=100_000,
            nrepeats=100,
        )
        fit_mcmc_mala.dump(fit_mal, dir_fit)
        print(fit_mal)
        assert fit_mal == fit_mcmc_mala.load(dir_fit)
        fit_mal = fit_mcmc_mala.load(dir_fit)
        nsamples = sum(fit_mal.yields)
        neff = mcmc.n_by_fit(fit_mal).sum()
        print(fit_mal.nsamples, neff, neff / fit_mal.nsamples)

    if 0:
        fit_ham = fit_mcmc_ham.fit(
            region_1,
            25,
            (0.0, 25.0),
            seed=0,
        )
        fit_mcmc_ham.dump(fit_ham, dir_fit)
        print(fit_ham)
        assert fit_ham == fit_mcmc_ham.load(dir_fit)
        nsamples = sum(fit_ham.yields)
        neff = mcmc.n_by_fit(fit_ham).sum()
        print(nsamples, neff, neff / nsamples)

        fit_nuts = fit_mcmc_nuts.fit(
            region_1,
            25,
            (0.0, 25.0),
            seed=0,
        )
        fit_mcmc_nuts.dump(fit_nuts, dir_fit)
        print(fit_nuts)
        assert fit_nuts == fit_mcmc_nuts.load(dir_fit)
        nsamples = sum(fit_nuts.yields)
        neff = mcmc.n_by_fit(fit_nuts).sum()
        print(nsamples, neff, neff / nsamples)

    if 1:
        fit_mal2 = fit_mymc_mala.fit(
            region_1,
            25,
            (0.0, 25.0),
            seed=0,
            nsamples=100,
            nrepeats=10,
            nprocesses=1,
        )
        fit_mymc_mala.dump(fit_mal2, dir_fit)
        print(fit_mal2)
        assert fit_mal2 == fit_mymc_mala.load(dir_fit)
        fit_mal = fit_mymc_mala.load(dir_fit)
        nsamples = sum(fit_mal2.yields)
        neff = mcmc.n_by_fit(fit_mal2).sum()
        print(fit_mal2.nsamples, neff, neff / fit_mal2.nsamples)


if __name__ == "__main__":
    main()
