"""
time python searches/ins1852821/test_fit.py

"""

import os

from pyhf_stuff import fit_cabinetry, fit_normal, region

BASEPATH = os.path.dirname(__file__)


def main():
    dir_region = os.path.join(BASEPATH, "SR0bvetotight")
    region_1 = region.load(dir_region)

    dir_fit = "fit"

    fit_cab = fit_cabinetry.fit(region_1)
    fit_cabinetry.dump(fit_cab, os.path.join(dir_region, dir_fit))
    print(fit_cab)
    print(fit_cabinetry.load(os.path.join(dir_region, dir_fit)))

    fit_norm = fit_normal.fit(region_1)
    fit_normal.dump(fit_norm, os.path.join(dir_region, dir_fit))
    print(fit_norm)
    print(fit_normal.load(os.path.join(dir_region, dir_fit)))
    # print(fit.filename(fit.interval), fit.interval(region_1))
    # print(fit.filename(fit.linspace), fit.linspace(region_1, 0, 4, 11))

    # print(
    # fit.filename(fit_mcmc),
    # fit_mcmc.mcmc_mala(region_1, 20, (0.0, 20.0), seed=0),
    # )


if __name__ == "__main__":
    main()
