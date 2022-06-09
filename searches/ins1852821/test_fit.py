import os

from pyhf_stuff import fit, region

BASEPATH = os.path.dirname(__file__)


def main():
    region_1 = region.load(os.path.join(BASEPATH, "SR0bvetotight"))

    test = fit.cabinetry_pre(region_1)
    print(test)
    test = fit.cabinetry_post(region_1)
    print(test)


if __name__ == "__main__":
    main()
