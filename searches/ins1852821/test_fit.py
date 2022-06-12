"""
time python searches/ins1852821/test_fit.py

"""

import os

from pyhf_stuff import (
    fit_cabinetry,
    fit_interval,
    fit_linspace,
    fit_normal,
    region,
)

BASEPATH = os.path.dirname(__file__)


def main():
    dir_region = os.path.join(BASEPATH, "SR0bvetotight")
    region_1 = region.load(dir_region)

    dir_fit = os.path.join(dir_region, "fit")

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

    # print(
    # fit.filename(fit_mcmc),
    # fit_mcmc.mcmc_mala(region_1, 20, (0.0, 20.0), seed=0),
    # )


if __name__ == "__main__":
    main()
