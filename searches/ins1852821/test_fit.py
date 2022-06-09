import os

from pyhf_stuff import fit, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_1 = region.load(os.path.join(BASEPATH, "SR0bvetotight"))

    print(fit.filename(fit.cabinetry_pre), fit.cabinetry_pre(region_1))
    print(fit.filename(fit.cabinetry_post), fit.cabinetry_post(region_1))
    print(fit.filename(fit.normal), fit.normal(region_1))
    print(fit.filename(fit.interval), fit.interval(region_1))


if __name__ == "__main__":
    main()
